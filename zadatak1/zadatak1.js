import { MongoClient } from "mongodb";
import { importData } from "./import.js";

const uri = "mongodb://admin:admin12345@localhost:27017/";
async function connectToMongoDB() {
  const client = new MongoClient(uri);
  await client.connect();
  return client;
}

/**
 * Zadatak 1: Zamjena nedostajućih vrijednosti
 * Replace missing values in the database:
 * - Continuous variables (numbers) with -1
 * - Categorical variables (strings) with "empty"
 */
async function replaceMissingValues(client) {
  console.log("\n" + "=".repeat(80));
  console.log("Zadatak 1: Zamjena nedostajućih vrijednosti");
  console.log("=".repeat(80) + "\n");

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
 * Zadatak 2: Izračun statistike za sve kontinuirane varijable
 * Calculate statistics for each continuous variable:
 * - Mean
 * - Standard deviation
 * - Count of non-missing values
 * And create a new document with these statistics
 */
async function calculateStatistics(client) {
  console.log("\n" + "=".repeat(80));
  console.log("Zadatak 2: Izračun statistike za sve kontinuirane varijable");
  console.log("=".repeat(80) + "\n");

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

  return statistics;
}

/**
 * Zadatak 3: Izračun frekvencija za kategoričke varijable
 * Za svaku kategoričku vrijednost izračunati frekvencije pojavnosti po obilježjima varijabli
 * i kreirati novi dokument koristeći nizove
 */
async function calculateFrequencies(client) {
  console.log("\n" + "=".repeat(80));
  console.log("Zadatak 3: Izračun frekvencija za kategoričke varijable");
  console.log("=".repeat(80) + "\n");

  console.log("Započinjem izračun frekvencija...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get the signals collection (where our data is stored)
  const signalsCollection = db.collection("signals");

  // Get a sample document to determine which fields are categorical (strings)
  const sampleDoc = await signalsCollection.findOne({});

  if (!sampleDoc) {
    console.log("Kolekcija signals je prazna, ne mogu izračunati frekvencije");
    return;
  }

  // Determine which fields are categorical (strings)
  const categoricalFields = [];

  for (const [field, value] of Object.entries(sampleDoc)) {
    if (typeof value === "string") {
      categoricalFields.push(field);
    }
  }

  console.log(
    `Pronađeno ${categoricalFields.length} kategoričkih polja za izračun frekvencija`
  );

  // If no categorical fields found, check metadata collection
  if (categoricalFields.length === 0) {
    console.log(
      "Nema kategoričkih polja u kolekciji signals, provjeravam kolekciju metadata..."
    );

    const metadataCollection = db.collection("metadata");
    const metadataDoc = await metadataCollection.findOne({});

    if (metadataDoc) {
      for (const [field, value] of Object.entries(metadataDoc)) {
        if (typeof value === "string") {
          categoricalFields.push(field);
        } else if (typeof value === "object" && value !== null) {
          // Check nested fields in metadata
          for (const [nestedField, nestedValue] of Object.entries(value)) {
            if (typeof nestedValue === "string") {
              categoricalFields.push(`${field}.${nestedField}`);
            }
          }
        }
      }

      console.log(
        `Pronađeno ${categoricalFields.length} kategoričkih polja u kolekciji metadata`
      );
    }
  }

  // Create frequencies object to store results
  const frequencies = {};

  // Calculate frequencies for each categorical field using $inc operator
  for (const field of categoricalFields) {
    console.log(`Izračunavam frekvencije za polje: ${field}`);

    // Use MongoDB aggregation to calculate frequencies
    const pipeline = [
      {
        $match: {
          [field]: { $ne: "empty" }, // Exclude our "empty" placeholder
        },
      },
      {
        $group: {
          _id: `$${field}`,
          count: { $sum: 1 },
        },
      },
      {
        $sort: { count: -1 },
      },
    ];

    const frequencyResults = await signalsCollection
      .aggregate(pipeline)
      .toArray();

    // Format the results as {value1: count1, value2: count2, ...}
    const fieldFrequencies = {};
    for (const result of frequencyResults) {
      fieldFrequencies[result._id] = result.count;
    }

    frequencies[field] = fieldFrequencies;

    console.log(
      `Frekvencije za ${field}: ${JSON.stringify(fieldFrequencies, null, 2)}`
    );
  }

  // Create a new collection for frequencies
  const freqCollection = db.collection("frekvencija_biosignals");

  // Create a document with the frequencies
  const freqDocument = {
    naziv: "frekvencija_biosignals",
    datumIzracuna: new Date(),
    frekvencije: frequencies,
  };

  // Insert the frequencies document
  await freqCollection.insertOne(freqDocument);

  console.log(
    "Frekvencije uspješno izračunate i pohranjene u kolekciju 'frekvencija_biosignals'"
  );

  return frequencies;
}

/**
 * Zadatak 4: Kreiranje dokumenata sa vrijednostima iznad i ispod srednje vrijednosti
 * Iz osnovnog dokumenta kreirati dva nova dokumenta sa kontinuiranim vrijednostima:
 * - Prvi dokument: elementi <= srednje vrijednosti
 * - Drugi dokument: elementi > srednje vrijednosti
 */
async function createStatisticalSplitDocuments(client, statistics) {
  console.log("\n" + "=".repeat(80));
  console.log(
    "Zadatak 4: Kreiranje dokumenata sa vrijednostima iznad i ispod srednje vrijednosti"
  );
  console.log("=".repeat(80) + "\n");

  console.log("Započinjem kreiranje dokumenata...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get the signals collection (where our data is stored)
  const signalsCollection = db.collection("signals");

  // Create new collections for the split data
  const belowAvgCollection = db.collection("statistika1_biosignals");
  const aboveAvgCollection = db.collection("statistika2_biosignals");

  // Clear existing data in these collections if they exist
  await belowAvgCollection.deleteMany({});
  await aboveAvgCollection.deleteMany({});

  console.log(
    "Obrisani postojeći dokumenti u kolekcijama statistika1_biosignals i statistika2_biosignals"
  );

  // Get all numeric fields and their mean values
  const numericFields = Object.keys(statistics);
  const meanValues = {};

  for (const field of numericFields) {
    meanValues[field] = statistics[field].srednjaVrijednost;
  }

  console.log("Srednje vrijednosti za polja:", meanValues);

  // Process documents in batches to avoid memory issues
  const batchSize = 1000;
  let processedCount = 0;
  let belowAvgCount = 0;
  let aboveAvgCount = 0;

  // Get total count for progress reporting
  const totalCount = await signalsCollection.countDocuments();
  console.log(`Ukupno dokumenata za obradu: ${totalCount}`);

  let cursor = signalsCollection.find({});
  let batch = [];

  while (await cursor.hasNext()) {
    const doc = await cursor.next();
    processedCount++;

    // Determine if document goes to below or above average collection
    let isBelowOrEqual = true;

    for (const field of numericFields) {
      if (doc[field] > meanValues[field]) {
        isBelowOrEqual = false;
        break;
      }
    }

    // Add to appropriate batch
    if (isBelowOrEqual) {
      await belowAvgCollection.insertOne(doc);
      belowAvgCount++;
    } else {
      await aboveAvgCollection.insertOne(doc);
      aboveAvgCount++;
    }

    // Log progress periodically
    if (processedCount % batchSize === 0 || processedCount === totalCount) {
      console.log(
        `Obrađeno ${processedCount}/${totalCount} dokumenata (${Math.round(
          (processedCount / totalCount) * 100
        )}%)`
      );
    }
  }

  console.log(`Završeno kreiranje dokumenata:`);
  console.log(
    `- Statistika1 (elementi <= srednje vrijednosti): ${belowAvgCount} dokumenata`
  );
  console.log(
    `- Statistika2 (elementi > srednje vrijednosti): ${aboveAvgCount} dokumenata`
  );
}

/**
 * Zadatak 5: Kopiranje osnovnog dokumenta i embedanje vrijednosti iz tablice frekvencija
 * Osnovni dokument kopirati u novi te embedati vrijednosti iz tablice 3 za svaku kategoričku vrijednost
 */
async function createEmbeddedFrequencyDocument(client, frequencies) {
  console.log("\n" + "=".repeat(80));
  console.log(
    "Zadatak 5: Kopiranje osnovnog dokumenta i embedanje vrijednosti frekvencija"
  );
  console.log("=".repeat(80) + "\n");

  console.log("Započinjem kreiranje dokumenta s embedanim frekvencijama...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get the signals collection (where our data is stored)
  const signalsCollection = db.collection("signals");

  // Create new collection for embedded document
  const embCollection = db.collection("emb_biosignals");

  // Clear existing data in this collection if it exists
  await embCollection.deleteMany({});

  console.log("Obrisani postojeći dokumenti u kolekciji emb_biosignals");

  // Process documents in batches to avoid memory issues
  const batchSize = 1000;
  let processedCount = 0;

  // Get total count for progress reporting
  const totalCount = await signalsCollection.countDocuments();
  console.log(`Ukupno dokumenata za obradu: ${totalCount}`);

  let cursor = signalsCollection.find({});
  let batch = [];

  while (await cursor.hasNext()) {
    const doc = await cursor.next();
    processedCount++;

    // Create a new document with all original fields
    const newDoc = { ...doc };

    // Add embedded frequency data for each categorical field
    for (const field in frequencies) {
      if (doc[field] && frequencies[field][doc[field]]) {
        newDoc[`${field}_frekvencija`] = frequencies[field][doc[field]];
      }
    }

    // Add to batch
    batch.push(newDoc);

    // Insert batch when it reaches the batch size
    if (batch.length >= batchSize) {
      await embCollection.insertMany(batch);
      batch = [];
      console.log(
        `Obrađeno ${processedCount}/${totalCount} dokumenata (${Math.round(
          (processedCount / totalCount) * 100
        )}%)`
      );
    }
  }

  // Insert any remaining documents
  if (batch.length > 0) {
    await embCollection.insertMany(batch);
    console.log(
      `Obrađeno ${processedCount}/${totalCount} dokumenata (${Math.round(
        (processedCount / totalCount) * 100
      )}%)`
    );
  }

  console.log(
    `Završeno kreiranje dokumenta s embedanim frekvencijama: ${processedCount} dokumenata`
  );
}

/**
 * Zadatak 6: Kopiranje osnovnog dokumenta i embedanje vrijednosti iz tablice statistike
 * Osnovni dokument kopirati u novi te embedati vrijednosti iz tablice 2 za svaku kontinuiranu vrijednost kao niz
 */
async function createEmbeddedStatisticsDocument(client, statistics) {
  console.log("\n" + "=".repeat(80));
  console.log(
    "Zadatak 6: Kopiranje osnovnog dokumenta i embedanje vrijednosti statistike"
  );
  console.log("=".repeat(80) + "\n");

  console.log("Započinjem kreiranje dokumenta s embedanim statistikama...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get the signals collection (where our data is stored)
  const signalsCollection = db.collection("signals");

  // Create new collection for embedded document
  const emb2Collection = db.collection("emb2_biosignals");

  // Clear existing data in this collection if it exists
  await emb2Collection.deleteMany({});

  console.log("Obrisani postojeći dokumenti u kolekciji emb2_biosignals");

  // Process documents in batches to avoid memory issues
  const batchSize = 1000;
  let processedCount = 0;

  // Get total count for progress reporting
  const totalCount = await signalsCollection.countDocuments();
  console.log(`Ukupno dokumenata za obradu: ${totalCount}`);

  let cursor = signalsCollection.find({});
  let batch = [];

  while (await cursor.hasNext()) {
    const doc = await cursor.next();
    processedCount++;

    // Create a new document with all original fields
    const newDoc = { ...doc };

    // Add embedded statistics data for each numeric field as an array
    for (const field in statistics) {
      if (doc[field] !== undefined) {
        newDoc[`${field}_statistika`] = [
          statistics[field].srednjaVrijednost,
          statistics[field].standardnaDevijacija,
          statistics[field].brojNomissingElemenata,
        ];
      }
    }

    // Add to batch
    batch.push(newDoc);

    // Insert batch when it reaches the batch size
    if (batch.length >= batchSize) {
      await emb2Collection.insertMany(batch);
      batch = [];
      console.log(
        `Obrađeno ${processedCount}/${totalCount} dokumenata (${Math.round(
          (processedCount / totalCount) * 100
        )}%)`
      );
    }
  }

  // Insert any remaining documents
  if (batch.length > 0) {
    await emb2Collection.insertMany(batch);
    console.log(
      `Obrađeno ${processedCount}/${totalCount} dokumenata (${Math.round(
        (processedCount / totalCount) * 100
      )}%)`
    );
  }

  console.log(
    `Završeno kreiranje dokumenta s embedanim statistikama: ${processedCount} dokumenata`
  );
}

/**
 * Zadatak 7: Izvlačenje srednjih vrijednosti s visokom standardnom devijacijom
 * Iz tablice emb2 izvući sve one srednje vrijednosti iz nizova čija je standardna devijacija 10% > srednje vrijednosti
 */
async function extractHighDeviationMeans(client) {
  console.log("\n" + "=".repeat(80));
  console.log(
    "Zadatak 7: Izvlačenje srednjih vrijednosti s visokom standardnom devijacijom"
  );
  console.log("=".repeat(80) + "\n");

  console.log("Započinjem izvlačenje srednjih vrijednosti...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get the emb2 collection
  const emb2Collection = db.collection("emb2_biosignals");

  // Get a sample document to find the statistic fields
  const sampleDoc = await emb2Collection.findOne({});

  if (!sampleDoc) {
    console.log(
      "Kolekcija emb2_biosignals je prazna, ne mogu izvući srednje vrijednosti"
    );
    return;
  }

  // Find all fields that end with "_statistika" (our embedded statistics arrays)
  const statisticFields = Object.keys(sampleDoc).filter((field) =>
    field.endsWith("_statistika")
  );

  console.log(`Pronađeno ${statisticFields.length} polja sa statistikama`);

  // Create a collection to store the results
  const highDevCollection = db.collection("visoka_devijacija_biosignals");

  // Clear existing data in this collection if it exists
  await highDevCollection.deleteMany({});

  // For each statistic field, check if standard deviation > 10% of mean
  const highDeviationStats = {};

  for (const field of statisticFields) {
    const originalField = field.replace("_statistika", "");

    // Get a document with non-null value for this field
    const doc = await emb2Collection.findOne({ [field]: { $ne: null } });

    if (doc && Array.isArray(doc[field]) && doc[field].length >= 3) {
      const mean = doc[field][0];
      const stdDev = doc[field][1];

      // Check if standard deviation is > 10% of mean
      if (stdDev > mean * 0.1) {
        highDeviationStats[originalField] = {
          srednjaVrijednost: mean,
          standardnaDevijacija: stdDev,
          omjer: (stdDev / mean).toFixed(4),
        };

        console.log(
          `Polje ${originalField} ima visoku devijaciju: srednja vrijednost = ${mean.toFixed(
            2
          )}, standardna devijacija = ${stdDev.toFixed(2)}, omjer = ${(
            stdDev / mean
          ).toFixed(4)}`
        );
      }
    }
  }

  // Store the results
  await highDevCollection.insertOne({
    naziv: "visoka_devijacija_biosignals",
    datumIzracuna: new Date(),
    polja: highDeviationStats,
  });

  console.log(
    `Završeno izvlačenje srednjih vrijednosti s visokom standardnom devijacijom: ${
      Object.keys(highDeviationStats).length
    } polja`
  );
}

/**
 * Zadatak 8: Kreiranje složenog indeksa na originalnoj tablici
 * Kreirati složeni indeks na originalnoj tablici i osmisliti upit koji je kompatibilan s indeksom
 */
async function createCompoundIndex(client) {
  console.log("\n" + "=".repeat(80));
  console.log("Zadatak 8: Kreiranje složenog indeksa na originalnoj tablici");
  console.log("=".repeat(80) + "\n");

  console.log("Započinjem kreiranje složenog indeksa...");

  // Get the biosignals database
  const db = client.db("biosignals");

  // Get the signals collection (original table)
  const signalsCollection = db.collection("signals");

  // Create a compound index on fields that are likely to be queried together
  // For biosignals data, we might want to query by sequence number and channel values
  const indexResult = await signalsCollection.createIndex(
    { nSeq: 1, CH1: 1, CH2: 1 },
    { name: "compound_nSeq_CH1_CH2" }
  );

  console.log(`Kreiran složeni indeks: ${indexResult}`);

  // Demonstrate a query that would use this index
  console.log("Izvršavam upit koji koristi složeni indeks...");

  const startTime = Date.now();

  const queryResult = await signalsCollection
    .find({
      nSeq: { $gte: 1000000, $lte: 2000000 },
      CH1: { $gte: 30000 },
      CH2: { $gte: 35000 },
    })
    .limit(10)
    .toArray();

  const endTime = Date.now();

  console.log(`Upit izvršen za ${endTime - startTime} ms`);
  console.log(
    `Pronađeno ${queryResult.length} dokumenata koji zadovoljavaju uvjete`
  );

  // Explain the query plan to verify index usage
  const explainResult = await signalsCollection
    .find({
      nSeq: { $gte: 1000000, $lte: 2000000 },
      CH1: { $gte: 30000 },
      CH2: { $gte: 35000 },
    })
    .explain();

  console.log("Složeni indeks uspješno kreiran i testiran");
}

async function main() {
  let client;
  try {
    console.log("\n" + "=".repeat(80));
    console.log("Započinjem import podataka iz data.txt datoteke");
    console.log("=".repeat(80) + "\n");

    // First import the data and wait for it to complete
    const importStats = await importData("data.txt");
    console.log("Import podataka završen sa statistikom:");
    console.log(`- Ukupno obrađenih linija: ${importStats.totalLines}`);
    console.log(`- Ukupno uvezenih zapisa: ${importStats.dataRecords}`);
    console.log(`- Uspješno: ${importStats.success ? "Da" : "Ne"}`);

    // Then connect to MongoDB and proceed with other tasks
    client = await connectToMongoDB();
    console.log("Uspješno povezan s MongoDB poslužiteljem");

    // Zadatak 1: Zamjena nedostajućih vrijednosti
    await replaceMissingValues(client);

    // Zadatak 2: Izračun statistike za kontinuirane varijable
    const statistics = await calculateStatistics(client);

    // Zadatak 3: Izračun frekvencija za kategoričke varijable
    const frequencies = await calculateFrequencies(client);

    // Zadatak 4: Kreiranje dokumenata sa vrijednostima iznad i ispod srednje vrijednosti
    await createStatisticalSplitDocuments(client, statistics);

    // Zadatak 5: Kopiranje osnovnog dokumenta i embedanje vrijednosti iz tablice frekvencija
    await createEmbeddedFrequencyDocument(client, frequencies);

    // Zadatak 6: Kopiranje osnovnog dokumenta i embedanje vrijednosti iz tablice statistike
    await createEmbeddedStatisticsDocument(client, statistics);

    // Zadatak 7: Izvlačenje srednjih vrijednosti s visokom standardnom devijacijom
    await extractHighDeviationMeans(client);

    // Zadatak 8: Kreiranje složenog indeksa na originalnoj tablici
    await createCompoundIndex(client);

    console.log("\n" + "=".repeat(80));
    console.log("Svi zadaci uspješno izvršeni!");
    console.log("=".repeat(80));
  } catch (err) {
    console.error("Greška pri izvršavanju zadataka:", err);
  } finally {
    if (client) {
      await client.close();
      console.log("MongoDB veza zatvorena");
    }
  }
}

main().catch(console.error);
