---
title:  "A basic Rust streaming ETL pipeline"
date:   2025-02-28 15:00:00
categories: [Tech, Rust, ETL] 
tags: [ETL, data engineering, streaming, rust]
---

Python is the go-to language for ETL pipelines, widely used in data engineering due to its simplicity, vast ecosystem (Pandas, Apache Airflow, PySpark), and rich community support. However, Python’s performance limitations, especially with high-throughput or real-time data processing, make it less than ideal for streaming-heavy workloads. Rust has been popping its head in the world of data engineering, and is becoming increasingly popular among data engineers. Rust's efficiency in memory usage and speed is particularly beneficial for tasks like data ingestion, where handling large volumes of data quickly is crucial. And in terms of crates, Rust has great libraries like [Polars](https://pola.rs/){:target="_blank"}, [DataFusion](https://datafusion.apache.org/){:target="_blank"}, [Arrow-rs](https://crates.io/crates/arrow){:target="_blank"}, [Ballista](https://crates.io/crates/ballista){:target="_blank"}, among many others.

Ironically, in this blog, I'll use none of those. I'll dial the complexity way down, and build a simple streaming ETL pipeline with Rust, using basic building blocks to achieve it. Most ETL pipelines in data engineering rely on batch processing, where data is collected, processed, and stored in discrete chunks. While effective for many use cases, batch processing introduces latency: each stage must wait for the previous one to finish before starting. This delay is problematic when dealing with high-throughput or real-time data streams. Streaming ETL, on the other hand, processes data continuously as it arrives, removing unnecessary waiting times. This approach is crucial for scenarios requiring near-instant insights, such as monitoring systems, fraud detection, or live analytics dashboards.

This pipeline reads a CSV file, does some basic processing, and stores the data into an SQLite database. We'll leverage async operations, efficient memory usage, and parallel processing. It ensures that no single step blocks the others—new data can be read and processed while previous batches are still being written to storage. I'll show the more interesting parts of the code, what makes it fast, and how to use it. The project utilizes several Rust libraries:
* [Tokio](https://crates.io/crates/tokio){:target="_blank"}: Provides an asynchronous runtime for handling I/O operations efficiently
* [Futures](https://crates.io/crates/futures){:target="_blank"}: Facilitates async streams for concurrent processing
* [CSV async](https://crates.io/crates/csv-async){:target="_blank"}: Asynchronously processes CSV files
* [SQLx](https://crates.io/crates/sqlx){:target="_blank"}: Enables asynchronous database interaction with SQLite
* [Clap](https://crates.io/crates/clap){:target="_blank"}: Command-line arguments parser

By structuring the pipeline as a series of concurrent tasks linked by in-memory channels, data flows seamlessly from ingestion to storage—without waiting for an entire batch to complete before moving forward. You'll be able to see the tool in action below. It clearly demonstrates how to create a new CSV file, or how to run the pipeline. As you can see, we process about 2.5 million lines in the CSV file in about 5 seconds, and 10 million lines in about 23 seconds. So let's how explore how to achieve this.
{% include youtube.html id="Y5vYKnsa3M4" %}

### CLI interface

I'm using the Clap crate as the command-line argument parser for this CLI. The following commands are defined:

``` rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "Rust ETL Pipeline")]
#[command(about = "A simple ETL pipeline that processes a CSV file and stores data in SQLite.")]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generates a CSV file with a given number of rows
    CreateFile {
        #[arg(long)]
        number_rows: u32,
    },
    /// Runs the ETL pipeline, processing the CSV and storing data in SQLite
    RunPipeline,
}
```
As you can see, we have 2 subcommands:
* CreateFile: Requires a number_rows argument to specify how many records should be generated in the CSV file.
* RunPipeline: Runs the entire ETL process, reading the CSV and storing data in the database.

The main function simply initializes some tracing, as well as the command parser. Depending on the users input, the respective function to either create a new CSV-file, or run the pipeline with that file as a data source, is launched.

``` rust
#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::new("rust_etl_pipeline=info,sqlx=error");
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    let cli = Args::parse();
    match cli.command {
        Some(Commands::CreateFile { number_rows }) => create_csv(number_rows).await?,
        Some(Commands::RunPipeline) => run_pipeline().await?,
        None => {
            println!("Run with --help to see instructions");
        }
    }

    Ok(())
}
```

### Creating a CSV file

Let's get the CSV creation out of the way. The code is quite straightforward. All we are doing is starting from a fixed array of first and last names, and combine randomly picked entries from both arrays.
``` rust
const FILE_NAME: &str = "./data/people.csv";
const FIRST_NAMES: [&str; 10] = [
    "Tom", "Johnny", "Jim", "Eric", "Amanda", "Grace", "Judy", "Frank", "Sally", "Will",
];
const LAST_NAMES: [&str; 10] = [
    "Connor",
    "Henderson",
    "Farley",
    "Henson",
    "Jeffries",
    "Carlin",
    "Anderson",
    "O' Sullivan",
    "Dorothy",
    "McDougal",
];
```

We will then write a random first name and last name, as wel as a random age, to a new line in the CSV file. However, we don't write the data line by line, but in bigger chunks. Writing to the CSV file line by line could look like this:
``` rust
for i in 0..number_rows {
    writer.write_all(format!("{},{},{}\n", first_name, last_name, age).as_bytes()).await?;
}
```

The problem with this, is that each .write_all() call would trigger a separate system call to handle the I/O operation. System calls are expensive because they involve switching from user space to kernel space. When writing a million rows, that equates to a million system calls, slowing down execution. Instead of writing each row separately, we will put multiple rows in a buffer, which is then written to disk when it's considered full. This greatly limits the amount of system calls, improving performance dramatically.

``` rust
let file = File::create(FILE_NAME).await?;
let mut writer = BufWriter::new(file);

// Write the header
writer.write_all(b"first_name,last_name,age\n").await?;

// We'll not write every single line, but write in chunks to limit the overhead
const CHUNK_SIZE: usize = 1000;
let mut buffer = String::with_capacity(CHUNK_SIZE * 50);

let mut rng = rand::rng();
for i in 0..number_rows {
    let first_name = FIRST_NAMES[rng.random_range(0..FIRST_NAMES.len())];
    let last_name = LAST_NAMES[rng.random_range(0..LAST_NAMES.len())];
    let age = rng.random_range(18..=65);

    buffer.push_str(&format!("{},{},{}\n", first_name, last_name, age));

    // Write the chunk if the chunk size is reached
    if i % CHUNK_SIZE as u32 == 0 {
        writer.write_all(buffer.as_bytes()).await?;
        buffer.clear();
    }
}

// Write any remaining data in the buffer.
if !buffer.is_empty() {
    writer.write_all(buffer.as_bytes()).await?;
}

writer.flush().await?;
```

### Running the pipeline

Let's look at the code that runs the actual pipeline. First, we'll create a new empty table. For simplicity, and for this blog's sake, we'll just drop the existing table and create it from scratch. This enables us to re-run the pipeline as much as we want. 

#### Prepare the database

``` rust
sqlx::query("DROP TABLE IF EXISTS people")
    .execute(pool)
    .await
    .context("Database error while dropping people table")?;

sqlx::query(
    r#"
CREATE TABLE IF NOT EXISTS people (
        id BLOB PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        age INTEGER,
        email TEXT)"#,
)
.execute(pool)
.await
.context("Database error while creating people table")?;

// Let's lower the synchronous mode to boost performance by reducing disk flushing
sqlx::query("PRAGMA synchronous = OFF")
    .execute(pool)
    .await
    .context("Database error while setting synchronous mode")?;

// Set a 40MB cache (negative value is KB)
sqlx::query("PRAGMA cache_size = -40000")
    .execute(pool)
    .await
    .context("Database error while setting cache size")?;
```

Now, one setting that requires a bit of explanation is the synchronous setting. For this test, we set it to OFF. What does that mean? The synchronous setting in SQLite modifies how the database engine handles synchronization operations, directly impacting data write performance and durability. It determines the frequency and method by which the database engine ensures that data is physically written to disk.

The possible values are:
* OFF: Disables synchronization, offering maximum performance but minimal data safety.
* NORMAL: Provides a balance between performance and durability by ensuring the database integrity is maintained, but without guaranteeing that all data is fully written to disk before completing each transaction.
* FULL: Ensures all data is fully written to disk before completing each transaction, providing maximum data integrity at the cost of slower write performance.
* EXTRA: An extension of FULL, offering additional durability guarantees.

By default, SQLite uses the FULL setting, which ensures that all data is fully written to disk before completing each transaction, providing maximum data integrity. Setting PRAGMA synchronous = OFF disables these synchronization operations, allowing SQLite to skip the steps that ensure data is safely written to disk. This can significantly improve write performance but at the cost of increased risk of data loss in the event of a system crash or power failure. 

The benefits are obvious: improved write performance. Disabling synchronization can lead to faster write operations, which is beneficial in scenarios where performance is critical, and data durability is less of a concern. Obviously this comes with a serious downside. Without synchronization, there is a higher risk of data corruption or loss during events like power failures or system crashes, as uncommitted transactions may not be fully written to disk.

When configuring the synchronous setting, it's essential to balance the need for performance with the requirement for data durability. For applications where data integrity is critical, retaining the default FULL setting or using NORMAL is advisable. For scenarios where performance is prioritized over data safety, such as processing transient data or performing bulk inserts where data can be regenerated if lost, setting synchronous = OFF may be acceptable.

#### Reading the CSV file

After the database is created, we'll read the CSV file, processes all lines concurrently, and inserts the processed data into the SQLite database in batches. We'll leverage asynchronous programming using Tokio, and channels to decouple the different stages (reading, processing, and writing). 

We first create 2 channels to connect the 3 concurrent tasks:
* One channel transfers raw CSV records from the reader task to the processor
* A second channel is responsible for transferring batches of processed data (i.e. Person structs) from the processor to the writer

We'll then spawn 3 separate asynchronous tasks using Tokio. We'll have a reader task which reads the CSV file concurrently and sends each line over the first channel. Using for_each_concurrent allows multiple records to be processed in parallel, which is ideal for I/O-bound operations such as reading from a file.

``` rust
// Create channels
let (to_workers, mut from_reader) = channel::<StringRecord>(100);
let (to_db, mut from_worker) = channel(100);

let reader_handle = tokio::spawn(async move {
    // Open the CSV file and concurrently process all records
    let file = File::open(FILE_NAME)
        .await
        .context("Couldn't open CSV file!")?;
    let mut reader = AsyncReader::from_reader(file);

    let num_workers = num_cpus::get();
    reader
        .records()
        .for_each_concurrent(num_workers, |record| {
            let to_workers = to_workers.clone();
            async move {
                let record = match record {
                    Ok(record) => record,
                    Err(e) => {
                        tracing::error!("Failed to parse CSV record: {:?}", e);
                        return;
                    }
                };
                to_workers.send(record).await.unwrap();
            }
        })
        .await;
});
```
#### Process each line

The processor task then receives each line, converts it to a Person object while adding computed fields like an email and capitalizing the last name, and batches them before sending them over the database channel. Batching objects like this instead of sending each created object over the channel separately has 2 benefits. Firstly, it reduces the traffic over the channel. This is definitely not the main reason for batching like this, but sending one message per batch can introduce overhead due to task wake-ups and context switching, which is always good to avoid. The main benefit from this batching: it greatly reduces the number of database insert operations, which will significantly improve performance when dealing with large datasets. 

``` rust
let processor_handle = tokio::spawn(async move {
    // Batch configuration
    const BATCH_SIZE: usize = 5_000;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    while let Some(record) = from_reader.recv().await {
        let first_name = &record[0];
        let last_name = &record[1];
        let age = record[2].parse().unwrap_or(0);
        let email = format!("{}.{}_{}@somemail.com", first_name, last_name, age);

        let person = Person::new(first_name, &last_name.to_uppercase(), age, &email);
        batch.push(person);

        if batch.len() == BATCH_SIZE {
            to_db.send(batch).await.unwrap();
            batch = Vec::with_capacity(BATCH_SIZE);
        }
    }

    // Send any remaining records
    if !batch.is_empty() {
        to_db.send(batch).await.unwrap();
    }
});
```

#### Write to the database

The last task receives all batches and performs a bulk insertion into the database. The SQL query string is built dynamically to include multiple value placeholders (one set per record). Values are collected in a vector and bound to the query one by one. Using a single dedicated connection for all write operations ensures that the inserts are executed in a controlled manner. At the end, we'll perform a VACUUM command which helps to optimize the database file after many inserts.

``` rust
let pool = pool.clone();
let writer_handle = tokio::spawn(async move {
    let start = std::time::Instant::now();

    // Dedicated connection for all writes
    let mut connection = pool
        .acquire()
        .await
        .context("Could not acquire database connection")?;

    while let Some(people) = from_worker.recv().await {
        let batch_length = people.len();

        let mut query =
            String::from("INSERT INTO people (id, first_name, last_name, age, email) VALUES ");
        let mut values = Vec::new();
        for (i, person) in people.iter().enumerate() {
            if i > 0 {
                query.push(',');
            }
            query.push_str("(?, ?, ?, ?, ?)");
            values.push(person.id.to_string());
            values.push(person.first_name.clone());
            values.push(person.last_name.clone());
            values.push(person.age.to_string());
            values.push(person.email.clone());
        }

        let mut query = sqlx::query(&query);
        for value in values {
            query = query.bind(value);
        }

        if let Err(e) = query.execute(&mut *connection).await {
            tracing::error!(
                "Failed to insert batch of {} records: {:?}",
                batch_length,
                e
            );

            // Skip to next batch
            continue;
        }
    }

    if let Err(e) = sqlx::query("VACUUM").execute(&pool).await {
        tracing::warn!("VACUUM failed: {:?}", e);
    }
});
```

### In conclusion

Building an ETL pipeline in Rust might seem daunting at first, especially when Python dominates the space with its rich ecosystem and simplicity. However, as this demonstrates, Rust’s performance advantages—especially in handling streaming data—are worth considering. By leveraging asynchronous programming with Tokio, efficient memory management, and concurrent processing, we’ve created a pipeline that not only processes millions of records in seconds but also does so in a structured and scalable way.

This basic implementation barely scratches the surface of what Rust can offer in data engineering. In real-world scenarios, you’d likely integrate more advanced libraries like Polars or Arrow for optimized data handling, or even push the boundaries with distributed processing using Ballista. But even in its simplest form, Rust proves to be a powerful alternative for high-performance ETL pipelines. If you’re a data engineer looking to experiment with Rust, this pipeline is a great starting point. Try tweaking it, optimizing it further, or even integrating it into a larger system. The potential is vast, and the ecosystem is only getting stronger.