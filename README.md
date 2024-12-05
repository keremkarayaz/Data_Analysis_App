# **Data Cleaning and Visualization Tool**

A multilingual web-based application designed to make **data cleaning** and **visualization** faster, smarter, and easier! This tool enables users to handle missing data, remove outliers, clean text, create visualizations, and much more, all from a user-friendly interface powered by **Streamlit**.

## **Features**

### 🎯 **Data Cleaning**
- **Fill Missing Values**: Choose from mean, median, or mode imputation.
- **Remove Duplicates**: Automatically remove duplicate rows.
- **Outlier Detection and Removal**: Uses IQR (Interquartile Range) to filter outliers.
- **Text Cleaning**: Strips whitespace, converts to lowercase, and standardizes text data.
- **Spelling Correction**: Automatically fixes common typos in text columns.
- **Format Correction**: Ensures numeric and datetime formats are accurate.
- **Binning**: Groups numeric columns into bins for analysis or visualization.

### 📊 **Data Visualization**
- **Histograms**: Visualize the distribution of numeric columns.
- **Boxplots**: Easily detect outliers and understand data spread.
- **Correlation Heatmaps**: Display relationships between numeric features.

### 🌐 **Multilingual Support**
- Available in **English** and **Türkçe**.
- Language selection available in the sidebar for easy switching.

### 📂 **File Handling**
- Supports **CSV**, **XLSX**, and **XLS** formats for uploading datasets.
- Cleaned datasets can be downloaded in **CSV format**.
- Visualizations can be downloaded as **PNG images**.

### ⚡ **Ease of Use**
- Interactive, no-code interface powered by **Streamlit**.
- Sidebar controls for all data cleaning and visualization options.
- Immediate feedback and preview for uploaded and cleaned datasets.

---

## **Tech Stack**
- **Python**: Core programming language.
- **Streamlit**: For building the web-based user interface.
- **Pandas**: Data manipulation and cleaning.
- **NumPy**: Numerical operations.
- **Seaborn** and **Matplotlib**: Data visualization.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/keremkarayaz/data-cleaning-tool.git
   cd data-cleaning-tool
   
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run app.py

## **Features**

- **🛠️ Data Cleaning Options**:
  - Fill missing values (mean, median, mode)
  - Remove duplicates
  - Remove outliers
  - Clean text data
  - Correct spelling errors
  - Correct data formats
  - Apply binning (grouping)

- **📊 Data Visualization Options**:
  - Histogram
  - Boxplot
  - Correlation heatmap

- **🌐 Multilingual Support**:
  - English and Turkish

- **📥 File Handling**:
  - Supports Excel (.xlsx, .xls) and CSV (.csv) formats
  - Download cleaned datasets
  - Save and download visualization images

---

## **Technologies Used**

- **Python**
- **Pandas** for data manipulation
- **NumPy** for numerical computations
- **Streamlit** for interactive web interface
- **Seaborn & Matplotlib** for data visualization
- **Scikit-learn** for preprocessing and handling missing values

---

## **Usage**

### **1. Upload your dataset**
Upload an Excel (.xlsx, .xls) or CSV file via the file uploader on the app.

### **2. Choose your cleaning options**
Use the sidebar to:
- Fill missing values
- Remove duplicates
- Remove outliers
- Clean text data
- Correct spelling errors
- Correct data formats
- Apply binning

### **3. Visualize your data**
Select visualization options like:
- Histogram
- Boxplot
- Correlation heatmap

### **4. Download the results**
Download the cleaned dataset and save any visualizations as PNG images.

---

## **Screenshots**

### **1. Home Page**
_Add a screenshot of your app's home page here._

### **2. Data Cleaning Options**
_Add a screenshot showing the sidebar with the cleaning options._

### **3. Visualizations**
_Add screenshots of the visualizations like histogram, boxplot, or heatmap._

---

## **Contributing**

### **How to contribute**
- Fork the repository
- Create a new branch (`git checkout -b feature/your-feature`)
- Commit your changes (`git commit -m "Add your message"`)
- Push to the branch (`git push origin feature/your-feature`)
- Open a pull request

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
