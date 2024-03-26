import pandas as pd
import numpy as np
import streamlit as st 
import altair as alt
from datetime import datetime
from mlxtend.frequent_patterns import fpgrowth, association_rules

class PreparationData:
    @staticmethod
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
    
    @staticmethod
    def is_data_cleaned_perfectly(df):
        missing_cells = df.isnull().sum().sum()
        missing_cells_percentage = (missing_cells / (len(df.columns) * len(df))) * 100
        duplicate_rows = df.duplicated().sum()
        duplicate_rows_percentage = (duplicate_rows / len(df)) * 100
        return missing_cells == 0 and missing_cells_percentage == 0 and duplicate_rows == 0 and duplicate_rows_percentage == 0
    
    @staticmethod
    def format_date(date_string):
        date_obj = datetime.strptime(date_string, '%m-%d-%Y')
        formatted_date = date_obj.strftime('%b %d, %Y')
        return formatted_date 
    
    @staticmethod
    def process_data(df):
        grouping_item = df.groupby(['customer', 'item_unit'])['item_name'].count().unstack().reset_index().fillna(0).set_index('customer')
        
        def one_hot_encoding(x):
            if x < 1:
                return 0
            if x > 0:
                return 1
            
        grouping_item = grouping_item.applymap(one_hot_encoding)
        return grouping_item

class ShowDataStatistics:
    @staticmethod
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
    
    @staticmethod
    def create_top_15_products_bar_chart(df):
        top_sold = (
            df.groupby('item_unit')['qty']
            .sum()
            .reset_index()
            .sort_values(by='qty', ascending=False)
            .head(15)
        )
        top_item_units = top_sold['item_unit'].tolist()

        chart = (
            alt.Chart(top_sold)
            .mark_bar()
            .encode(
                x=alt.X('item_unit:N', title='Top Item Units', sort='-y'),
                y=alt.Y('qty:Q', title='Total Quantity Sold'),
                tooltip=[
                    alt.Tooltip('item_unit:N', title='Item Unit'),
                    alt.Tooltip('qty:Q', title='Total Quantity Sold')
                ],
                color=alt.condition(
                    alt.datum.rank < 4,
                    alt.value('green'),
                    alt.value('steelblue')
                )
            )
            .properties(
                width=600,
                height=400,
                title='Top 15 Products Sold by Item'
            )
            .transform_window(
                rank='rank(qty)',
                sort=[alt.SortField('qty', order='descending')]
            )
        )

        st.altair_chart(chart, use_container_width=True)

        if top_item_units:
            st.info(f"The items unit **{', '.join(top_item_units[:1])}**, **{', '.join(top_item_units[1:2])}**, and **{', '.join(top_item_units[2:3])}** are the top 3 most purchased.")
    
    @staticmethod
    def create_line_chart_gmv(df):
        df_retail = df[df['customer'] == 'UMUM/CASH']
        gmv_by_date_retail = df_retail.groupby(['order_month_day'])['total_price'].sum().reset_index()
        
        df_member = df[df['customer'] != 'UMUM/CASH']
        gmv_by_date_member = df_member.groupby(['order_month_day'])['total_price'].sum().reset_index()
        
        combined_df = pd.concat([gmv_by_date_retail.assign(customer='Retail'), gmv_by_date_member.assign(customer='Member')])
        
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
        max_gmv_retail_date_formatted = PreparationData.format_date(max_gmv_retail_date)
        max_gmv_retail_price = '{:,.0f}'.format(gmv_by_date_retail['total_price'].max())

        min_gmv_retail_date = gmv_by_date_retail.loc[gmv_by_date_retail['total_price'].idxmin()]['order_month_day']
        min_gmv_retail_date_formatted = PreparationData.format_date(min_gmv_retail_date)
        min_gmv_retail_price = '{:,.0f}'.format(gmv_by_date_retail['total_price'].min())

        # Find the date with the highest and lowest GMV for Member
        max_gmv_member_date = gmv_by_date_member.loc[gmv_by_date_member['total_price'].idxmax()]['order_month_day']
        max_gmv_member_date_formatted = PreparationData.format_date(max_gmv_member_date)
        max_gmv_member_price = '{:,.0f}'.format(gmv_by_date_member['total_price'].max())

        min_gmv_member_date = gmv_by_date_member.loc[gmv_by_date_member['total_price'].idxmin()]['order_month_day']
        min_gmv_member_date_formatted = PreparationData.format_date(min_gmv_member_date)
        min_gmv_member_price = '{:,.0f}'.format(gmv_by_date_member['total_price'].min())

        st.info(f"On the **Member**, **the peak** Gross Merchandise Value (GMV) was recorded on **{max_gmv_member_date_formatted}**, reaching **{max_gmv_member_price}**, while the **lowest** GMV occurred on **{min_gmv_member_date_formatted}** at **{min_gmv_member_price}**.\n\n In the **Retail** customer type, the **highest** GMV was observed on **{max_gmv_retail_date_formatted}**, totaling **{max_gmv_retail_price}**, whereas the **lowest** GMV was noted on **{min_gmv_retail_date_formatted}**, at **{min_gmv_retail_price}**.")
        
    @staticmethod
    def create_pie_chart(df):
        num_retail = (df['customer'] == 'UMUM/CASH').sum()
        num_member = (df['customer'] != 'UMUM/CASH').sum()

        total_customers = num_retail + num_member
        percent_retail = round((num_retail / total_customers) * 100, 2)
        percent_member = round((num_member / total_customers) * 100, 2)

        pie_data = pd.DataFrame({
            'Customer Type': ['Retail', 'Member'],
            'Percentage': [percent_retail, percent_member]
        })

        pie_chart = (
            alt.Chart(pie_data)
            .mark_arc()
            .encode(
                color=alt.Color('Customer Type:N', scale=alt.Scale(domain=['Retail', 'Member'], range=['green', 'blue'])),
                angle=alt.Angle('Percentage:Q'),
                tooltip=['Customer Type:N', 'Percentage:Q']
            )
            .properties(
                width=400,
                height=400,
                title='Percentage of Customer Types'
            )
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    
        st.info(f"The percentage based on the buyer type from **Retail** is **{percent_retail}%**, while **Member** is **{percent_member}%**.")