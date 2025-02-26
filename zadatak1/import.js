const fs = require("fs");
const { MongoClient } = require("mongodb");
const readline = require("readline");

// MongoDB connection string
const uri = "mongodb://admin:admin12345@localhost:27017/";
const client = new MongoClient(uri);

// Function to read data from file and import to MongoDB
async function importData() {
  try {
    // Connect to MongoDB
    await client.connect();
    console.log("Connected to MongoDB");

    // Create or use database
    const database = client.db("biosignals");
    const metadataCollection = database.collection("metadata");
    const signalsCollection = database.collection("signals");

    // Create a read stream for the data file
    const fileStream = fs.createReadStream("data.txt");
    const rl = readline.createInterface({
      input: fileStream,
      crlfDelay: Infinity,
    });

    let metadata = null;
    let columnNames = [];
    let isHeader = true;
    let documents = [];
    let lineCount = 0;
    let dataLineCount = 0;

    // Process each line
    for await (const line of rl) {
      lineCount++;

      // Process header section
      if (isHeader) {
        if (line.startsWith("# OpenSignals")) {
          console.log("Found OpenSignals header");
          continue;
        } else if (line.startsWith("# {")) {
          // Extract JSON metadata from the comment line
          try {
            const jsonStr = line.substring(2); // Remove the "# " prefix
            metadata = JSON.parse(jsonStr);
            console.log("Parsed metadata successfully");

            // Extract column names from metadata
            const deviceId = Object.keys(metadata)[0];
            columnNames = metadata[deviceId].column;
            console.log("Column names:", columnNames);

            // Save metadata to MongoDB
            await metadataCollection.insertOne(metadata);
            console.log("Saved metadata to MongoDB");
          } catch (err) {
            console.error("Error parsing metadata:", err);
          }
          continue;
        } else if (line.startsWith("# EndOfHeader")) {
          isHeader = false;
          console.log("End of header section, starting to process data");
          continue;
        } else if (line.startsWith("#")) {
          // Skip other comment lines
          continue;
        }
      }

      // Process data lines
      if (!isHeader) {
        try {
          // Split the tab-separated values
          const values = line.split("\t");

          // Create a document with named fields
          const document = {};

          // Add each column with its name
          for (let i = 0; i < values.length && i < columnNames.length; i++) {
            // Convert to number if possible
            const numValue = Number(values[i]);
            document[columnNames[i]] = isNaN(numValue) ? values[i] : numValue;
          }

          documents.push(document);
          dataLineCount++;

          // Insert in batches of 1000 for better performance
          if (documents.length >= 1000) {
            await signalsCollection.insertMany(documents);
            console.log(`Inserted ${documents.length} signal records`);
            documents = [];
          }
        } catch (err) {
          console.error(`Error processing data line ${lineCount}: ${line}`);
          console.error(err);
        }
      }
    }

    // Insert any remaining documents
    if (documents.length > 0) {
      await signalsCollection.insertMany(documents);
      console.log(`Inserted final ${documents.length} signal records`);
    }

    console.log(`Total lines processed: ${lineCount}`);
    console.log(`Total data records imported: ${dataLineCount}`);
    console.log("Data import completed");
  } catch (err) {
    console.error("Error during import:", err);
  } finally {
    await client.close();
  }
}

// Run the import function
importData().catch(console.error);
