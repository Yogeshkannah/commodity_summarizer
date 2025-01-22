# Commodity Summarizer

The tool's purpose is to deliver concise summaries for each agricultural commodity class, offering a quick and simplified overview of complex commodity data. Its name implies providing users with clear insights at a glance, streamlining their understanding.

**Reference**
- [Get sample PDF](https://www.usda.gov/oce/commodity/wasde)

**Prerequisites**
- [Python 3.10+](https://www.python.org/downloads/#:~:text=Python%203.10.15,Release%20Notes)
- [PineCone](https://github.com/Antony-M1/opensearch_docker) - get the pinecone documentation link.
- [Flake 8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) for the PEP 8 Standard Edition

# .env
```py
GOOGLE_API_KEY=<YOUR_API_KEY>
PINE_CONE_API=<YOUR_API_KEY>
```
load the api keys to your environment

# Quick Start
### Step 1: Clone the repo
```
git clone https://github.com/Yogeshkannah/commodity_summarizer.git
```
### Step 2: Create & Activate environment
```
python -m venv venv
```
```
source venv/bin/activate
```

### Step 3: Install dependencies
```
pip install -r requirements.txt
```

### Step 4: Run the project
```
streamlit run main.py
```