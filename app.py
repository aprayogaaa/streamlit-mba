import streamlit as st  
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from streamlit_option_menu import option_menu
from mlxtend.frequent_patterns import fpgrowth, association_rules
from myclass import PreparationData, ShowDataStatistics

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

        if st.button("Upload New File"):
            st.session_state['data']['df'] = None
            st.experimental_rerun()
            
        st.subheader("Dataset Statistics:")
        statistics_df = ShowDataStatistics.calculate_dataset_statistics(st.session_state['data']['df'])
        st.write(statistics_df)
        
        if PreparationData.is_data_cleaned_perfectly(st.session_state['data']['df']):
            st.success("Data is cleaned perfectly!")
        else:
            st.warning("Data set not cleaned perfectly!")

    else:
        upload_file = st.file_uploader('Upload an Excel file', type=['xlsx', 'xls'])

        if upload_file is not None:
            df = pd.read_excel(upload_file, engine='openpyxl')
            df = PreparationData.clean_and_transform_data(df)
            st.dataframe(df)
            
            if st.button("Upload New File"):
                st.session_state['data']['df'] = None
                st.experimental_rerun()
            
            st.subheader("Dataset Statistics:")
            statistics_df = ShowDataStatistics.calculate_dataset_statistics(df)
            st.write(statistics_df)
            
            if PreparationData.is_data_cleaned_perfectly(df):
                st.success("Data is cleaned perfectly!")
            else:
                st.warning("Data set not cleaned perfectly!")

            st.session_state['data']['df'] = df
        else:
            st.write("You didn't upload a new file!")

if selected == 'Dashboard':
    st.title(f'Dashboard')
    if st.session_state['data']['df'] is not None:
        df = st.session_state['data']['df']
        ShowDataStatistics.create_top_15_products_bar_chart(df)
        ShowDataStatistics.create_line_chart_gmv(df)
        ShowDataStatistics.create_pie_chart(df)
    else:
        "Please, upload your file first!"
     
if selected == 'Generate Bundle':
    st.title(f'Generate Bundle')
    if st.session_state['data']['df'] is not None:
        df = st.session_state['data']['df']
        grouping_item_result = PreparationData.process_data(df)

        support_helper = '''
                            > If a product's support value exceeds the minimum support value, it can be recommended as a commonly purchased product.
        '''
        
        confidence_helper = '''
                            > The minimum confidence value is used to validate the correlation between two or more products. The greater the minimum confidence value, the stronger the association or correlation.
                            In this application, the minimum confidence is set at 50% or higher.
        '''
        
        min_support = st.slider("Minimum Support", min_value=0.1, max_value=0.9, value=0.20, help=support_helper)
        min_confidence = st.slider("Minimum Confidence", min_value=0.5, max_value=1.0, value=0.6, help=confidence_helper)

        fpgrowth_model = fpgrowth(grouping_item_result.astype('bool'), min_support=min_support, use_colnames=True)

        if not fpgrowth_model.empty:
            association_rules_result = association_rules(fpgrowth_model, metric="lift", min_threshold=1.0)          
            bundling_options = sorted(association_rules_result['antecedents'].apply(len).unique())
            display_bundling_options = [option + 1 for option in bundling_options]
            selected_bundling_options = st.multiselect("Select Bundling Possibilities:", display_bundling_options)
            selected_bundling_options = [option - 1 for option in selected_bundling_options]

            filtered_rules = association_rules_result[
                (association_rules_result['support'] >= min_support) &
                (association_rules_result['confidence'] >= min_confidence) &
                (association_rules_result['antecedents'].apply(len).isin(selected_bundling_options))
            ]

            st.markdown("## Frequent Itemsets")
            st.info("Frequent itemsets are products that are frequently purchased together in transaction data.")
            
            frequent_itemsets_str = [f"item {i + 1}: {', '.join(map(str, item))}" for i, (item, support) in enumerate(zip(fpgrowth_model['itemsets'], fpgrowth_model['support']))]
            
            st.write(frequent_itemsets_str)
            st.markdown("## Association Rules")
            st.info("Association rules refer to the relationships or associations between items in a data transanction that frequently occur together.")
            
            association_rules_str = [
                f"Rule: {list(rule['antecedents'])} ==> {list(rule['consequents'])}"
                for i, rule in filtered_rules.iterrows()
            ]
            st.write(association_rules_str)
        else:
            st.write("No frequent itemsets or association rules found with the specified minimum support or minimum confidence.")
    else:
        "Please, upload your file first!"






