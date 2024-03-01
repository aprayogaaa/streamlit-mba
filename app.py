import streamlit as st  
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from streamlit_option_menu import option_menu
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Function to clean and transform the data
def clean_and_transform_data(df):
    df = df[df['qty'] != 0]
    df['qty'] = df['qty'].apply(lambda x: round(x))
    unit_mapping = {
        '1/2': 'KG',
        '1/4': 'KG',
        'STG': 'KG',
        'BOK': 'BOX',
        'RTN': 'RTG',
        'TPL': 'TPLS',
        'SLOP': 'PAK',
        'SLP': 'PAK',
        'KRG': 'SAK',
        'KLN': 'BTL',
        'LBR': 'LMBR'
    }
    df['unit'] = df['unit'].apply(lambda unit: unit_mapping.get(unit, unit))
    df['customer'] = df['customer'].replace('CUSTEMER', 'PELANGGAN BAROKAH')
    df['item_unit'] = df['item_name'].str.cat(df['unit'], sep='-')
    df['order_month_day'] = df['date'].dt.strftime('%m-%d-%Y')
    return df

# Function to calculate dataset
def calculate_dataset_statistics(df):
    num_variables = len(df.columns)
    num_observations = len(df)
    num_categorical_variables = len(df.select_dtypes(include=['object']).columns)
    num_numeric_variables = len(df.select_dtypes(include=[np.number]).columns)
    missing_cells = df.isnull().sum().sum()
    missing_cells_percentage = (missing_cells / (num_variables * num_observations)) * 100
    duplicate_rows = df.duplicated().sum()
    duplicate_rows_percentage = (duplicate_rows / num_observations) * 100

    statistics_data = {
        'Metric': ['Number of Columns', 'Number of Rows', 'Number of Categorical Columns',
                   'Number of Numeric Columns', 'Missing Cells', 'Missing Cells Percentage (%)',
                   'Duplicate Rows', 'Duplicate Rows Percentage (%)'],
        'Value': [num_variables, num_observations, num_categorical_variables,
                  num_numeric_variables, missing_cells, missing_cells_percentage,
                  duplicate_rows, duplicate_rows_percentage]
    }

    return pd.DataFrame(statistics_data)

# Function to check data is cleaned perfectly!
def is_data_cleaned_perfectly(df):
    missing_cells = df.isnull().sum().sum()
    missing_cells_percentage = (missing_cells / (len(df.columns) * len(df))) * 100
    duplicate_rows = df.duplicated().sum()
    duplicate_rows_percentage = (duplicate_rows / len(df)) * 100

    return missing_cells == 0 and missing_cells_percentage == 0 and duplicate_rows == 0 and duplicate_rows_percentage == 0

# Function to create a bar chart of the top 15 products sold
def create_top_15_products_bar_chart(df):
    top_sold = df.groupby('item_unit')['qty'].sum().reset_index().sort_values(by='qty', ascending=False).head(15)
    top_item_units = top_sold['item_unit'].tolist()
    
    chart = alt.Chart(top_sold).mark_bar().encode(
        x=alt.X('item_unit:N', title='Top Item Units', sort='-y'),
        y=alt.Y('qty:Q', title='Total Quantity Sold'),
        tooltip=[alt.Tooltip('item_unit:N', title='Item Unit'), alt.Tooltip('qty:Q', title='Total Quantity Sold')],
        color=alt.condition(
            alt.datum.rank < 4,
            alt.value('green'),
            alt.value('steelblue')
        )
    ).properties(
        width=600,
        height=400,
        title='Top 15 Products Sold by Item'
    ).transform_window(
        rank='rank(qty)',
        sort=[alt.SortField('qty', order='descending')]
    )

    st.altair_chart(chart, use_container_width=True)

    if top_item_units:
        st.info(f"The items unit **{', '.join(top_item_units[:1])}**, **{', '.join(top_item_units[1:2])}**, and **{', '.join(top_item_units[2:3])}** are the top 3 most purchased.")

 
# Function to format date
def format_date(date_string):
    date_obj = datetime.strptime(date_string, '%m-%d-%Y')
    formatted_date = date_obj.strftime('%b %d, %Y')
    return formatted_date 
        
# Function to create a line chart GMV
def create_line_chart_gmv(df):
    df_retail = df[df['customer'] == 'UMUM/CASH']
    gmv_by_date_retail = df_retail.groupby(['order_month_day'])['total_price'].sum().reset_index()
    
    df_member = df[df['customer'] != 'UMUM/CASH']
    gmv_by_date_member = df_member.groupby(['order_month_day'])['total_price'].sum().reset_index()
    
    # Combine the dataframes
    combined_df = pd.concat([gmv_by_date_retail.assign(customer='Retail'), gmv_by_date_member.assign(customer='Member')])
    
    # Create a line chart
    chart = alt.Chart(combined_df).mark_line().encode(
        x=alt.X('order_month_day:T', title='Purchased Date'),
        y=alt.Y('total_price:Q', title='Total Price'),
        color=alt.Color('customer:N', scale=alt.Scale(domain=['Retail', 'Member'], range=['green', 'blue']), title='Customer Type'),
        tooltip=[
            alt.Tooltip('order_month_day:T', title='Purchased Date'),
            alt.Tooltip('total_price:Q', title='Total Price', format=',.0f')
        ]
    ).properties(
        width=600,
        height=400,
        title='Gross Merchandise Value (GMV) Over Time'
    )

    st.altair_chart(chart, use_container_width=True)

    # Find the date with the highest and lowest GMV for Retail
    max_gmv_retail_date = gmv_by_date_retail.loc[gmv_by_date_retail['total_price'].idxmax()]['order_month_day']
    max_gmv_retail_date_formatted = format_date(max_gmv_retail_date)
    max_gmv_retail_price = '{:,.0f}'.format(gmv_by_date_retail['total_price'].max())

    min_gmv_retail_date = gmv_by_date_retail.loc[gmv_by_date_retail['total_price'].idxmin()]['order_month_day']
    min_gmv_retail_date_formatted = format_date(min_gmv_retail_date)
    min_gmv_retail_price = '{:,.0f}'.format(gmv_by_date_retail['total_price'].min())

    # Find the date with the highest and lowest GMV for Member
    max_gmv_member_date = gmv_by_date_member.loc[gmv_by_date_member['total_price'].idxmax()]['order_month_day']
    max_gmv_member_date_formatted = format_date(max_gmv_member_date)
    max_gmv_member_price = '{:,.0f}'.format(gmv_by_date_member['total_price'].max())

    min_gmv_member_date = gmv_by_date_member.loc[gmv_by_date_member['total_price'].idxmin()]['order_month_day']
    min_gmv_member_date_formatted = format_date(min_gmv_member_date)
    min_gmv_member_price = '{:,.0f}'.format(gmv_by_date_member['total_price'].min())

    st.info(f"On the **Member**, **the peak** Gross Merchandise Value (GMV) was recorded on **{max_gmv_member_date_formatted}**, reaching **{max_gmv_member_price}**, while the **lowest** GMV occurred on **{min_gmv_member_date_formatted}** at **{min_gmv_member_price}**.\n\n In the **Retail** customer type, the **highest** GMV was observed on **{max_gmv_retail_date_formatted}**, totaling **{max_gmv_retail_price}**, whereas the **lowest** GMV was noted on **{min_gmv_retail_date_formatted}**, at **{min_gmv_retail_price}**.")    


# Function to create a pie chart
def create_pie_chart(df):
    # Calculate the number of Retail and Member customers
    num_retail = (df['customer'] == 'UMUM/CASH').sum()
    num_member = (df['customer'] != 'UMUM/CASH').sum()

    # Calculate the percentage of each customer type
    total_customers = num_retail + num_member
    percent_retail = (num_retail / total_customers) * 100
    percent_member = (num_member / total_customers) * 100

    # Create a DataFrame for the pie chart
    pie_data = pd.DataFrame({
        'Customer Type': ['Retail', 'Member'],
        'Percentage': [percent_retail, percent_member]
    })

    # Create the pie chart using Altair
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        color=alt.Color('Customer Type:N', scale=alt.Scale(domain=['Retail', 'Member'], range=['green', 'blue'])),
        tooltip=['Customer Type:N', 'Percentage:Q'],
        angle=alt.Angle('Percentage:Q'),
    ).properties(
        width=400,
        height=400,
        title='Percentage of Customer Types'
    )

    st.altair_chart(pie_chart, use_container_width=True)

# Function to encode the data
def process_data(df):
    grouping_item = df.groupby(['customer', 'item_unit'])['item_name'].count().unstack().reset_index().fillna(0).set_index('customer')
    def one_hot_encoding(x):
        if x < 1:
            return 0
        if x > 0:
            return 1
    grouping_item = grouping_item.applymap(one_hot_encoding)
    return grouping_item

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Upload File', 'Dashboard', 'Generate Bundle'],
        icons=['house-door-fill', 'cloud-arrow-up-fill', 'bar-chart-fill', 'basket-fill'],
        menu_icon='cast',
        default_index=0
    )

# Inisialisasi st.session_state['data']
if 'data' not in st.session_state:
    st.session_state['data'] = {'df': None}

if selected == 'Home':
    st.title('✨Welcome to The Bundle App✨')
    st.markdown('''
                ---
                ### 🪶About The Application
                 This application is the implementation of an market basket analysis. The purpose of this application is to help business owners make product bundles to increase their sales. Additionally, this application provides a dashboard to give business owners more insights into their data, including gross merchandise value (GMV), top products sold, and customer type percentages.
                 
                 ### 🛒What is Market Basket Analysis
                 Market basket analysis (MBA) is a data analysis technique that analyzes sets of frequently occurring objects (frequent itemsets) in a dataset and the relationships or correlations between items in that dataset. This analysis technique can generate insights that can be used to build marketing and advertising strategies.
                 
                 Ref: *Han, J., Pei, J., & Tong, H. (2022). Data Mining: Concepts and Techniques (4th ed.). Morgan Kaufmann.*
                 
                 ### ☎️Contact Us
                 If you have any inquires, don't hesitate to contact us below:
                 
                 - 👦 Agung Prayoga (agung.prayoga002@binus.ac.id)
                 - 👧 Natasya Kinata (natasya.kinata@binus.ac.id)
                 - 👦 Christopher (christopher020@binus.ac.id)
                ''')

if selected == 'Upload File':
    st.title('Upload Your File')
    st.markdown('''
                    Before upload a file, make sure that you follow the data template [here](https://docs.google.com/spreadsheets/d/1-ixWeXlCcizvAPUmpLeOsh6FSbKTAWpd58EH33t23Z8/edit?usp=sharing).
                ''')

    if st.session_state['data']['df'] is not None:
        st.write("Data already uploaded:")
        st.dataframe(st.session_state['data']['df'])

        # Button "Upload File" to change the data
        if st.button("Upload New File"):
            st.session_state['data']['df'] = None
            st.experimental_rerun()
            
        # Display dataset statistics
        st.subheader("Dataset Statistics:")
        statistics_df = calculate_dataset_statistics(st.session_state['data']['df'])
        st.write(statistics_df)
        
        # Display data cleaning information
        if is_data_cleaned_perfectly(st.session_state['data']['df']):
            st.success("Data is cleaned perfectly!")
        else:
            st.warning("Data set not cleaned perfectly!")

    else:
        upload_file = st.file_uploader('Upload an Excel file', type=['xlsx', 'xls'])

        if upload_file is not None:
            df = pd.read_excel(upload_file, engine='openpyxl')
            # Clean and transform the data
            df = clean_and_transform_data(df)
            st.dataframe(df)

            # Button "Upload New File" to change the data
            if st.button("Upload New File"):
                st.session_state['data']['df'] = None
                st.experimental_rerun()
            
            # Display dataset statistics
            st.subheader("Dataset Statistics:")
            statistics_df = calculate_dataset_statistics(df)
            st.write(statistics_df)
            
            # Display data cleaning information
            if is_data_cleaned_perfectly(df):
                st.success("Data is cleaned perfectly!")
            else:
                st.warning("Data set not cleaned perfectly!")

            # Save the cleaned DataFrame to st.session_state
            st.session_state['data']['df'] = df
        else:
            st.write("You didn't upload a new file!")

if selected == 'Dashboard':
    st.title(f'Dashboard')
    if st.session_state['data']['df'] is not None:
        df = st.session_state['data']['df']
        create_top_15_products_bar_chart(df)
        create_line_chart_gmv(df)
    else:
        "Please, upload your file first!"
     
if selected == 'Generate Bundle':
    st.title(f'Generate Bundle')
    if st.session_state['data']['df'] is not None:
        df = st.session_state['data']['df']
        
        # Process the data into FP-Growth model
        grouping_item_result = process_data(df)

        # Create helper to explain support and confidene
        support_helper = '''
                            > If a product's support value exceeds the minimum support value, it can be recommended as a commonly purchased product.
        '''
        
        confidence_helper = '''
                            > The minimum confidence value is used to validate the correlation between two or more products. The greater the minimum confidence value, the stronger the association or correlation.
                            In this application, the minimum confidence is set at 50% or higher.
        '''
        
        # Sliders for minimum support and minimum confidence
        min_support = st.slider("Minimum Support", min_value=0.0, max_value=1.0, value=0.20, help=support_helper)
        min_confidence = st.slider("Minimum Confidence", min_value=0.5, max_value=1.0, value=0.6, help=confidence_helper)

        # Build FP-Growth model
        fpgrowth_model = fpgrowth(grouping_item_result, min_support=min_support, use_colnames=True)
        
        print(fpgrowth_model)

        if not fpgrowth_model.empty:
            # Build association rules
            association_rules_result = association_rules(fpgrowth_model, metric="lift", min_threshold=1.0)

            print(association_rules_result)
            
            # Get unique bundling options from association rules and increment by 1
            bundling_options = sorted(association_rules_result['antecedents'].apply(len).unique())
            display_bundling_options = [option + 1 for option in bundling_options]

            print(display_bundling_options)
            
            # Multiselect widget for bundling options
            selected_bundling_options = st.multiselect("Select Bundling Possibilities:", display_bundling_options)

            # Map back to the original bundling options for filtering
            selected_bundling_options = [option - 1 for option in selected_bundling_options]

            # Filter association rules by minimum support, minimum confidence, and selected bundling options
            filtered_rules = association_rules_result[
                (association_rules_result['support'] >= min_support) &
                (association_rules_result['confidence'] >= min_confidence) &
                (association_rules_result['antecedents'].apply(len).isin(selected_bundling_options))
            ]

            print(filtered_rules)    
                
            # Display the frequent itemsets
            st.markdown("## Frequent Itemsets")
            st.info("Frequent itemsets are products that are frequently purchased together in transaction data.")
            
            frequent_itemsets_str = [f"item {i + 1}: {', '.join(map(str, item))}" for i, (item, support) in enumerate(zip(fpgrowth_model['itemsets'], fpgrowth_model['support']))]
            st.write(frequent_itemsets_str)

            print(frequent_itemsets_str)
            
            # Display the unique association rules
            st.markdown("## Association Rules")
            st.info("Association rules refer to the relationships or associations between items in a data transanction that frequently occur together.")
            
            association_rules_str = [
                f"Rule: {list(rule['antecedents'])} ==> {list(rule['consequents'])}"
                for i, rule in filtered_rules.iterrows()
            ]
            st.write(association_rules_str)

            print(association_rules_str)
            
        else:
            st.write("No frequent itemsets or association rules found with the specified minimum support or minimum confidence.")
    else:
        "Please, upload your file first!"






