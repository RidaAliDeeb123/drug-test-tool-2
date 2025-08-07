# Gender-Biased Drug Response Risk Detector

A machine learning-based tool to detect and analyze gender bias in drug response, helping healthcare providers make more informed decisions.

## Features

- Universal data ingestion from CSV, EHR, and manual entry
- Gender bias detection in datasets
- BioClinicalBERT integration for drug mention extraction
- Interactive visualization dashboard
- Clinical decision support
- Model transparency and bias reporting

## Installation

### Using Docker (Recommended)

```bash
docker build -t drug-gender-bias .
docker run -p 8501:8501 drug-gender-bias
```

### Local Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Access the application through your web browser at `http://localhost:8501`
2. Choose your data input method:
   - CSV Upload: Upload a CSV file with patient data
   - Manual Entry: Enter patient data manually
   - EHR Data: Integrate with EHR systems (coming soon)
3. The application will automatically:
   - Detect gender bias in the dataset
   - Extract drug mentions from clinical notes
   - Generate visualizations
   - Provide risk assessment

## Data Format

The application supports the following columns in input data:
- `gender`: Patient gender (male/female)
- `age`: Patient age
- `drug`: Drug name
- `dosage`: Drug dosage
- `clinical_notes`: Unstructured clinical notes

## Model Architecture

The system uses a hybrid approach combining:
- BioClinicalBERT for NLP tasks
- TabPFN for structured data prediction
- Custom bias detection algorithms

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
