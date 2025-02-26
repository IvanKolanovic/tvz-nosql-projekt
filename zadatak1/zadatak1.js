import { MongoClient } from "mongodb";

const uri = "mongodb://admin:admin12345@localhost:27017/";

async function connectToMongoDB() {
  const client = new MongoClient(uri);
  await client.connect();
  return client;
}

/**
 * Replace missing values in the database:
 * - Continuous variables (numbers) with -1
 * - Categorical variables (strings) with "empty"
 */
async function replaceMissingValues(client) {
  console.log("Zadatak 1: Zamjena nedostajućih vrijednosti \n");

  console.log("Započinjem zamjenu nedostajućih vrijednosti...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get all collections in the database
  const collections = await db.listCollections().toArray();

  for (const collectionInfo of collections) {
    const collectionName = collectionInfo.name;
    const collection = db.collection(collectionName);

    console.log(`Obrađujem kolekciju: ${collectionName}`);

    // Get a sample document to determine field types
    const sampleDoc = await collection.findOne({});

    if (!sampleDoc) {
      console.log(`Kolekcija ${collectionName} je prazna, preskačem`);
      continue;
    }

    // Determine which fields are numeric and which are categorical
    const numericFields = [];
    const categoricalFields = [];

    for (const [field, value] of Object.entries(sampleDoc)) {
      if (typeof value === "number") {
        numericFields.push(field);
      } else if (typeof value === "string") {
        categoricalFields.push(field);
      }
    }

    console.log(
      `Pronađeno ${numericFields.length} numeričkih polja i ${categoricalFields.length} kategoričkih polja`
    );

    // Replace missing values in numeric fields with -1
    if (numericFields.length > 0) {
      const numericUpdates = {};
      numericFields.forEach((field) => {
        numericUpdates[field] = { $eq: null };
      });

      const numericResult = await collection.updateMany(
        { $or: numericFields.map((field) => ({ [field]: null })) },
        { $set: Object.fromEntries(numericFields.map((field) => [field, -1])) }
      );

      console.log(
        `Ažurirano ${numericResult.modifiedCount} dokumenata s nedostajućim numeričkim vrijednostima`
      );
    }

    // Replace missing values in categorical fields with "empty"
    if (categoricalFields.length > 0) {
      const categoricalResult = await collection.updateMany(
        { $or: categoricalFields.map((field) => ({ [field]: null })) },
        {
          $set: Object.fromEntries(
            categoricalFields.map((field) => [field, "empty"])
          ),
        }
      );

      console.log(
        `Ažurirano ${categoricalResult.modifiedCount} dokumenata s nedostajućim kategoričkim vrijednostima`
      );
    }
  }

  console.log("Završena zamjena nedostajućih vrijednosti");
}

/**
 * Calculate statistics for each continuous variable:
 * - Mean
 * - Standard deviation
 * - Count of non-missing values
 * And create a new document with these statistics
 */
async function calculateStatistics(client) {
  console.log("Zadatak 2: Izračun statistike za sve kontinuirane varijable \n");
  console.log("Započinjem izračun statistike...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get the signals collection (where our data is stored)
  const signalsCollection = db.collection("signals");

  // Get a sample document to determine which fields are numeric
  const sampleDoc = await signalsCollection.findOne({});

  if (!sampleDoc) {
    console.log("Kolekcija signals je prazna, ne mogu izračunati statistiku");
    return;
  }

  // Determine which fields are numeric (continuous variables)
  const numericFields = [];

  for (const [field, value] of Object.entries(sampleDoc)) {
    if (typeof value === "number") {
      numericFields.push(field);
    }
  }

  console.log(
    `Pronađeno ${numericFields.length} numeričkih polja za statistiku`
  );

  // Create statistics object to store results
  const statistics = {};

  // Calculate statistics for each numeric field
  for (const field of numericFields) {
    console.log(`Izračunavam statistiku za polje: ${field}`);

    // Get all non-missing values for this field
    // Note: We're excluding documents where the field is -1 (our missing value indicator)
    const values = await signalsCollection
      .find({ [field]: { $ne: -1 } })
      .project({ [field]: 1 })
      .toArray();

    // Extract just the values into an array
    const fieldValues = values.map((doc) => doc[field]);

    // Count of non-missing values
    const count = fieldValues.length;

    if (count === 0) {
      console.log(`Nema nomissing vrijednosti za polje ${field}`);
      statistics[field] = {
        srednjaVrijednost: null,
        standardnaDevijacija: null,
        brojNomissingElemenata: 0,
      };
      continue;
    }

    // Calculate mean
    const sum = fieldValues.reduce((acc, val) => acc + val, 0);
    const mean = sum / count;

    // Calculate standard deviation
    const squaredDifferences = fieldValues.map((val) =>
      Math.pow(val - mean, 2)
    );
    const variance =
      squaredDifferences.reduce((acc, val) => acc + val, 0) / count;
    const stdDev = Math.sqrt(variance);

    // Store statistics for this field
    statistics[field] = {
      srednjaVrijednost: mean,
      standardnaDevijacija: stdDev,
      brojNomissingElemenata: count,
    };

    console.log(
      `Statistika za ${field}: srednja vrijednost = ${mean.toFixed(
        2
      )}, standardna devijacija = ${stdDev.toFixed(
        2
      )}, broj nomissing elemenata = ${count}`
    );
  }

  // Create a new collection for statistics
  const statsCollection = db.collection("statistika_biosignals");

  // Create a document with the statistics
  const statsDocument = {
    naziv: "statistika_biosignals",
    datumIzracuna: new Date(),
    statistike: statistics,
  };

  // Insert the statistics document
  await statsCollection.insertOne(statsDocument);

  console.log(
    "Statistika uspješno izračunata i pohranjena u kolekciju 'statistika_biosignals'"
  );
}

async function main() {
  let client;
  try {
    client = await connectToMongoDB();
    console.log("Uspješno povezan s MongoDB poslužiteljem");

    // Call the function to replace missing values
    await replaceMissingValues(client);

    // Call the function to calculate statistics
    await calculateStatistics(client);
  } catch (err) {
    console.error("Greška pri povezivanju s MongoDB-om:", err);
  } finally {
    if (client) {
      await client.close();
      console.log("MongoDB veza zatvorena");
    }
  }
}

main().catch(console.error);
