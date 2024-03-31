# End-To-End Market Basket Analysis (MBA) Application Using Streamlit
## Description

Market Basket Analysis (MBA) is a data mining technique used to uncover associations between items purchased together in transactions. This project aims to implement MBA using Streamlit, a Python library for creating web applications, to provide an interactive and user-friendly interface for analyzing transaction data. The data used in this application is based on data from Toko Barokah, a grocery store that was founded in 1998 and is located in Indramayu, West Java. Currently, Toko Barokah is just a small shop that sells some food and drinks.

## Algorithm
![runtime apriori vs fp-growth](https://github.com/aprayogaaa/streamlit-mba/assets/70948216/150911b4-1ac5-438e-8bdc-2992f23a328d)

Figure above, based on Toko Barokah transaction, shows the execution time of the Apriori and FP-Growth algorithms. The Apriori algorithm works faster in processing small data, as seen in the graph with high minimum support values, while the FP-Growth algorithm works faster when processing larger data, as determined by low minimum support values. Additionally, the difference in execution time with high minimum support values for both algorithms is almost the same. Therefore, based on the execution time results of both algorithms, FP-Growth is a suitable model for implementation into a website using Streamlit.

## Features
- Interactive dashboard for visualizing top products, gross merchandise value (GMV), and percetage of customer type
- Support for uploading transaction data
- Customizable parameters for association rule mining algorithms based on support and confidence value
- Customizable bundling product

## Files
- `app.py`: The main Python script containing the Streamlit application code for market basket analysis.
- `myclass.py`: Contains of data preparation and visualization.
- `requirements.txt`: A list of Python dependencies required for running the application. Install these dependencies using `pip install -r requirements.txt`.

## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/aprayogaaa/streamlit-mba.git
```

2. Navigate into the project directory
```bash
cd streamlit-mba
```

3. Install dependency
```bash
pip install -r requirements.txt
```

## Usage
1. Sreamlit run app.py.
```bash
streamlit run app.py
```

2. Follow the link the localhost.
3. pload your transaction data in Excel format (must follow the data template provided).
4. Adjust the parameters for association rule mining (e.g., support and confidence).
5. Explore the generated association rules and visualizations.

### Contributing
If you'd like to contribute to this project, please follow these guidelines:

- Fork the repository.
- Create a new branch `git checkout -b feature`.
- Make your changes.
- Commit your changes `git commit -am 'Add new feature`.
- Push to the branch `git push origin feature`.
- Create a new Pull Request.



  
