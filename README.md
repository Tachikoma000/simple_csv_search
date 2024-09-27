# **Simple CSV Search**

A Python-based tool for performing semantic searches on CSV files using OpenAI embeddings and FAISS for efficient similarity search.

---

## **Table of Contents**

- [**Simple CSV Search**](#simple-csv-search)
  - [**Table of Contents**](#table-of-contents)
  - [**Introduction**](#introduction)
  - [**Features**](#features)
  - [**How It Works**](#how-it-works)
    - [**Embeddings**](#embeddings)
    - [**Embedding Models**](#embedding-models)
    - [**Search Mechanism**](#search-mechanism)
  - [**Installation**](#installation)
    - [**Prerequisites**](#prerequisites)
    - [**Steps**](#steps)
  - [**Usage**](#usage)
    - [**Configuration**](#configuration)
    - [**Running the Application**](#running-the-application)
  - [**Project Structure**](#project-structure)
  - [**Examples**](#examples)
    - [**Sample Data Entry**](#sample-data-entry)
    - [**Performing a Search**](#performing-a-search)
  - [**Notes**](#notes)
  - [**Contributing**](#contributing)
  - [**License**](#license)
  - [**Detailed Explanation**](#detailed-explanation)
    - [**Embeddings**](#embeddings-1)
    - [**Embedding Models**](#embedding-models-1)
    - [**Search Mechanism**](#search-mechanism-1)
    - [**Why This Approach Works**](#why-this-approach-works)

---

## **Introduction**

**Simple CSV Search** is a tool that allows you to perform semantic searches over data stored in CSV files. By leveraging OpenAI's embedding models and FAISS (Facebook AI Similarity Search), the application can find and retrieve data entries that are semantically similar to a user's query, even if exact keywords do not match.

This project is ideal for:

- Searching large CSV datasets.
- Implementing intelligent search features in data analysis tools.
- Learning how to integrate OpenAI embeddings with vector similarity search.

---

## **Features**

- **Semantic Search**: Find relevant data entries based on the meaning of your query, not just keyword matching.
- **Scalable**: Handles large datasets efficiently using FAISS for vector indexing.
- **Customizable**: Easily adapt the code to work with different datasets and search requirements.
- **Interactive Command-Line Interface**: Simple interface for entering queries and viewing results.

---

## **How It Works**

### **Embeddings**

Embeddings are numerical vector representations of text data that capture the semantic meaning of the text. In this application:

- **Data Embeddings**: Each row in the CSV file is converted into a text string by concatenating all columns. This text is then transformed into an embedding using an OpenAI embedding model.
- **Query Embeddings**: User queries are also converted into embeddings using the same model.
- **Semantic Space**: Both data and query embeddings exist in the same high-dimensional space, allowing for meaningful comparison based on their content.

### **Embedding Models**

OpenAI provides powerful embedding models that can transform text into embeddings:

- **Models Used**: The application uses models like `text-embedding-3-small` or `text-embedding-3-large`.
- **Model Selection**: You can choose the model in the `config.py` file based on your needs for performance and accuracy.
- **API Integration**: The embeddings are obtained via OpenAI's API, requiring an API key.

### **Search Mechanism**

The search functionality relies on comparing the query embedding with the data embeddings:

- **Vector Similarity Search**: Uses FAISS to perform efficient similarity searches in high-dimensional space.
- **Distance Metric**: By default, the application uses Euclidean distance (L2) to measure similarity. A smaller distance indicates higher similarity.
- **Top-K Results**: The application retrieves the top `k` most similar data entries to the user's query.

---

## **Installation**

### **Prerequisites**

- **Python 3.7+**
- **pip** package manager
- **OpenAI API Key**: You need an API key from OpenAI to use their embedding models.

### **Steps**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/simple_csv_search.git
   cd simple_csv_search
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - On **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - On **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Configuration**

   - Copy `config_example.py` to `config.py`.

     ```bash
     cp config_example.py config.py
     ```

   - Open `config.py` and add your OpenAI API key and adjust settings as needed.

5. **Prepare Your CSV File**

   - Place your CSV file (e.g., `data.csv`) in the project directory.
   - Ensure the CSV file has headers and that all columns you want to include are properly named.

---

## **Usage**

### **Configuration**

Before running the application, ensure your `config.py` file is correctly set up:

```python
# config.py

# OpenAI API Key Configuration
OPENAI_API_KEY = 'your-openai-api-key'  # Replace with your actual API key

# OpenAI Embedding Model
EMBEDDING_MODEL = 'text-embedding-3-small'  # or 'text-embedding-3-large'

# CSV File Path
CSV_FILE_PATH = 'data.csv'  # Ensure your CSV file is in the project directory

# Text Column Name in CSV
TEXT_COLUMN = 'text'  # The column name after processing (usually 'text')

# FAISS Index File Path
INDEX_FILE_PATH = 'faiss_index.index'  # Path to save/load the FAISS index

# Number of Neighbors to Retrieve
TOP_K = 5  # Number of top results to return in a search
```

### **Running the Application**

```bash
python app.py
```

**Steps:**

1. **Generating Embeddings**

   - Upon running, the application will generate embeddings for your data.
   - This may take some time depending on the size of your dataset.
   - Progress messages will be displayed in the console.

2. **Entering Search Queries**

   - After embeddings are generated, you'll be prompted to enter your search query.
   - Type a query relevant to your data and press Enter.

     ```
     Enter your search query (or type 'exit' to quit): Your query here
     ```

3. **Viewing Results**

   - The application will display the top matching results along with their similarity scores.
   - Example output:

     ```
     Top Results:
     Score: 0.0000
     Text: stationId: 3 | pollutant: SO2 | value: 3 | date: 2024-09-27 | hour: 14

     Score: 2.3456
     Text: stationId: 3 | pollutant: NO2 | value: 1 | date: 2024-09-27 | hour: 14
     ```

4. **Exiting the Application**

   - To exit, type `exit` at the query prompt.

     ```
     Enter your search query (or type 'exit' to quit): exit
     ```

---

## **Project Structure**

```
simple_csv_search/
├── app.py
├── config_example.py
├── config.py          # (Excluded from version control)
├── data_processor.py
├── embedder.py
├── search_engine.py
├── requirements.txt
├── data.csv           # Your CSV data file
├── .gitignore
└── README.md
```

- **`app.py`**: Main application script.
- **`config.py`**: Configuration file containing API keys and settings (should be kept secret).
- **`config_example.py`**: Example configuration file without sensitive information.
- **`data_processor.py`**: Handles loading and preprocessing of CSV data.
- **`embedder.py`**: Generates embeddings using OpenAI's API.
- **`search_engine.py`**: Builds and manages the FAISS index for searching.
- **`requirements.txt`**: Lists Python package dependencies.
- **`.gitignore`**: Specifies files to exclude from version control.
- **`README.md`**: Documentation for the project.

---

## **Examples**

### **Sample Data Entry**

Suppose your CSV file contains environmental data like pollutant levels:

| stationId | pollutant | value |     date     | hour |
|-----------|-----------|-------|--------------|------|
|     3     |    SO2    |   3   | 2024-09-27   |  14  |
|     3     |     O3    |  12   | 2024-09-27   |  14  |
|     3     |    NO2    |   1   | 2024-09-27   |  14  |

### **Performing a Search**

**Query:**

```
Enter your search query (or type 'exit' to quit): High levels of SO2 on 2024-09-27
```

**Output:**

```
Top Results:
Score: 0.0000
Text: stationId: 3 | pollutant: SO2 | value: 3 | date: 2024-09-27 | hour: 14

Score: 2.1234
Text: stationId: 3 | pollutant: O3 | value: 12 | date: 2024-09-27 | hour: 14

Score: 3.5678
Text: stationId: 3 | pollutant: NO2 | value: 1 | date: 2024-09-27 | hour: 14
```

**Explanation:**

- The application found that the first entry is most relevant to the query about SO2 levels on the specified date.
- The scores represent the Euclidean distance between the query embedding and the data embeddings; lower scores indicate higher similarity.

---

## **Notes**

- **API Costs**: Generating embeddings via OpenAI's API may incur costs depending on the number of tokens processed.
- **Data Privacy**: Ensure that your data complies with OpenAI's usage policies and data privacy regulations.
- **Performance**: For very large datasets, consider implementing additional optimizations, such as saving and loading the FAISS index to avoid regenerating embeddings.

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Your detailed description of the changes."
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request**

---

## **License**

This project is licensed under the MIT License.

---

## **Detailed Explanation**

### **Embeddings**

**What Are Embeddings?**

- Embeddings are vectors of floating-point numbers that represent the semantic meaning of text.
- They are generated by neural network models trained to capture linguistic patterns and relationships.

**Why Use Embeddings?**

- **Semantic Similarity**: They allow for measuring how semantically similar two pieces of text are.
- **Dimensionality Reduction**: They convert high-dimensional textual data into fixed-size numerical vectors that are easier to process.

**How Embeddings Are Generated in This Application:**

1. **Text Preparation**: Each row of the CSV is converted into a text string by concatenating all columns.
2. **API Request**: The text is sent to OpenAI's API, which returns an embedding vector.
3. **Batch Processing**: To improve efficiency, texts are processed in batches.

**Example:**

```python
# In embedder.py

def get_embedding(text, model=EMBEDDING_MODEL):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']
```

### **Embedding Models**

**OpenAI's Embedding Models:**

- **`text-embedding-3-small`**: A smaller model suitable for general-purpose embeddings with lower computational costs.
- **`text-embedding-3-large`**: A larger model that may provide better performance on certain tasks.

**Choosing a Model:**

- **Trade-offs**: Larger models may offer better embeddings but require more computational resources and may have higher API costs.
- **Configuration**: Set the model in `config.py`.

**Model Specifications:**

- **Input Limits**: Models have maximum token limits (e.g., 8191 tokens).
- **Output Vector Size**: The dimensionality of the embeddings (e.g., 1536 for `text-embedding-3-small`).

### **Search Mechanism**

**FAISS (Facebook AI Similarity Search):**

- An open-source library for efficient similarity search on dense vectors.
- Capable of handling large-scale datasets with millions of vectors.

**How Search Works in This Application:**

1. **Building the Index:**

   - Embeddings are added to a FAISS index (`IndexFlatL2`), which uses L2 (Euclidean) distance.
   - This index enables quick retrieval of vectors similar to a query vector.

2. **Processing a Query:**

   - The user's query is converted into an embedding.
   - The FAISS index searches for the nearest neighbors to the query embedding.

3. **Retrieving Results:**

   - The application retrieves the top `k` results with the smallest distances to the query embedding.
   - These results are presented to the user along with their similarity scores.

**Similarity Metrics:**

- **Euclidean Distance (L2)**: Measures the straight-line distance between two points in vector space.
- **Interpretation**: A lower distance indicates higher similarity between embeddings.

**Example:**

```python
# In search_engine.py

def search(self, query_embedding, top_k=TOP_K):
    distances, indices = self.index.search(np.array([query_embedding]), top_k)
    return distances[0], indices[0]
```

### **Why This Approach Works**

- **Semantic Understanding**: Embeddings capture the meaning behind text, allowing for more nuanced search results.
- **Efficient Retrieval**: FAISS enables quick searches even in large datasets by optimizing vector computations.
- **Flexible Queries**: Users can input natural language queries without needing to match exact keywords.
