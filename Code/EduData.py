import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import geopandas as gpd
import censusdata
import pydeck as pdk

sns.set_theme(style="whitegrid")

# Function to load data from multiple sources
def load_data():
    """
    Load datasets related to employment, wages, and skills from CSV files.
    """
    employment_projections = pd.read_csv("data/1 Employment Projections by Industry Sheet1.csv", header=1)
    nc_occupational_employment_wages = pd.read_csv("data/2_NC_Occupational_Employment_and_Wages(cleaned).csv")
    clean_cte = pd.read_csv("data/3_4_cleanCTE2.csv")
    fifth_cleaned = pd.read_csv("data/5th_cleaned.csv")
    nc_occupational_outcomes = pd.read_csv("data/6 NC Occupational Outcomes 1 Year After Grad.xlsx - Sheet1.csv", header=1)
    nc_post_grad_enrollment = pd.read_csv("data/7 NC Post Grad Enrollment After 1 Year.csv", header=1)
    industry_skills_needs = pd.read_csv("data/industry_skills_needs.csv")
    return employment_projections, nc_occupational_employment_wages, clean_cte, fifth_cleaned, nc_occupational_outcomes, nc_post_grad_enrollment, industry_skills_needs

# Function to preprocess the employment projections dataset
def preprocess_df1(employment_projections):
    """
    Preprocess the employment projections dataframe by replacing '*' with NaN,
    converting string to numeric values, and handling percentage values.
    """
    employment_projections.replace('*', np.nan, inplace=True)
    # Convert columns to numeric, handling commas, dollar signs, and percentages
    columns_to_convert = ['2021', '2030', 'Net Growth', 'Average Weekly', 'Annualized']
    for column in columns_to_convert:
        if column in ['Average Weekly', 'Annualized']:
            employment_projections[column] = pd.to_numeric(employment_projections[column].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', ''), errors='coerce') / (100 if column == 'Annualized' else 1)
        else:
            employment_projections[column] = pd.to_numeric(employment_projections[column].astype(str).str.replace(',', ''), errors='coerce')
    return employment_projections

# Plotting functions to visualize different aspects of the datasets
def plot_average_weekly_wages(df):
    """
    Plot the distribution of average weekly wages.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Average Weekly'].dropna(), kde=True, color="cornflowerblue")
    plt.title('Distribution of Average Weekly Wages')
    plt.xlabel('Average Weekly Wage')
    plt.ylabel('Frequency')
    st.pyplot(plt)

def plot_top_20_net_growth(df):
    """
    Plot the top 20 industries by net growth.
    """
    top_20_net_growth = df.sort_values(by='Net Growth', ascending=False).head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Net Growth', y='Industry Title', data=top_20_net_growth, color="cornflowerblue")
    plt.title('Top 20 Industries by Net Growth')
    plt.xlabel('Net Growth')
    plt.ylabel('Industry Title')
    plt.tight_layout()
    st.pyplot(plt)

def plot_top_n_net_growth_in(df):
    """
    Plot the top N industries by net growth based on user selection.
    """
    n = st.sidebar.slider('Select the number of top industries', min_value=5, max_value=30, value=20)
    top_n_net_growth = df.sort_values(by='Net Growth', ascending=False).head(n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Net Growth', y='Industry Title', data=top_n_net_growth, color="cornflowerblue")
    plt.title(f'Top {n} Industries by Net Growth')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('Net Growth')
    plt.xlabel('Industry Title')
    plt.tight_layout()
    st.pyplot(plt)

def interactive_avg_weekly_wages(df):
    """
    Generates an interactive histogram of average weekly wages within a user-selected range.

    Parameters:
    - df: DataFrame containing the column 'Average Weekly' with wage data.

    The function utilizes Streamlit's sidebar slider for range selection and seaborn for plotting.
    """
    # Determine the minimum and maximum average weekly wages for the slider bounds
    min_wage, max_wage = df['Average Weekly'].min(), df['Average Weekly'].max()

    # Create a slider in the Streamlit sidebar for selecting the wage range
    selected_range = st.sidebar.slider(
        'Select Average Weekly Wage Range',
        min_value=int(min_wage),
        max_value=int(max_wage),
        value=(int(min_wage), int(max_wage))
    )

    # Filter the DataFrame to include only the wages within the selected range
    filtered_df = df[(df['Average Weekly'] >= selected_range[0]) & (df['Average Weekly'] <= selected_range[1])]

    # Plot the distribution of average weekly wages within the selected range
    plt.figure(figsize=(12, 6))
    sns.histplot(filtered_df['Average Weekly'], kde=True)
    plt.title('Distribution of Average Weekly Wages Within Selected Range')
    plt.xlabel('Average Weekly Wage')
    plt.ylabel('Frequency')

    # Display the plot in the Streamlit app
    st.pyplot(plt)


def show_bubble_chart(df):
    # Filter out total occupations and convert columns to numeric
    df_filtered = df[df['Occupation'] != 'Total, All Occupations'].copy()
    df_filtered['Employment'] = pd.to_numeric(df_filtered['Employment'].str.replace(',', ''), errors='coerce')
    df_filtered['Annual wage; mean'] = pd.to_numeric(df_filtered['Annual wage; mean'].str.replace('[\$,]', '', regex=True), errors='coerce')

    # Streamlit widgets for user interaction
    sort_criteria = st.sidebar.selectbox('Sort by:', ['Employment', 'Annual wage; mean'])
    num_occupations = st.sidebar.slider('Number of Occupations to Display', min_value=5, max_value=50, value=25)

    # Sort and select top N occupations based on selected criteria
    top_occupations = df_filtered.nlargest(num_occupations, sort_criteria)

    # Bubble chart
    plt.figure(figsize=(14, 10))
    bubble_chart = sns.scatterplot(
        data=top_occupations, 
        y='Occupation', 
        x='Employment', 
        size='Annual wage; mean', 
        sizes=(100, 2000),  # Adjust the range of bubble sizes as needed
        alpha=0.5
    )
    plt.title(f'Top {num_occupations} Occupations by {sort_criteria} (Bubble Chart)')
    plt.xlabel('Employment')
    plt.ylabel('Occupation')
    plt.xticks(rotation=90)
    bubble_chart.legend(
        title='Annual Wage Mean',
        loc='center left', bbox_to_anchor=(1, 0.8),
        fontsize='large', title_fontsize='x-large',
        handlelength=4, handleheight=2, labelspacing=2
    )
    plt.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    

   

def interactive_employment_outcomes_by_industry(df):
    """
    Interactive visualization to display employment outcomes by industry over time.
    Allows users to select specific industries for comparison.
    """
    # Convert relevant columns to float, replacing commas
    df[df.columns[4:]] = df[df.columns[4:]].replace(',', '', regex=True).astype(float)
    
    # Sidebar multiselect for choosing industries to view
    industries = st.sidebar.multiselect('Select Industries to View', options=df.columns[4:], default=df.columns[4])
    
    # Plot settings
    plt.figure(figsize=(14, 8))
    for industry in industries:
        plt.plot(df['Year'], df[industry], marker='o', label=industry)
        plt.text(df['Year'].iloc[-1], df[industry].iloc[-1], industry, ha='left', va='center')
    plt.title('Employment Outcomes by Industry Over Time')
    plt.xlabel('Year')
    plt.ylabel('Employment Outcomes')
    plt.xticks(df['Year'])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

def interactive_employment_outcomes_by_industry(df):
    # Replace commas and convert columns 4 onwards to float
    df[df.columns[4:]] = df[df.columns[4:]].replace(',', '', regex=True).astype(float)
    
    # Allow user to select specific industries to view
    industries = st.sidebar.multiselect('Select Industries', options=df.columns[4:], default=df.columns[4])
    
    plt.figure(figsize=(14, 8))
    
    for column in industries:
        plt.plot(df['Year'], df[column], marker='o', label=column)
        plt.text(df['Year'].iloc[-1], df[column].iloc[-1], column, ha='left', va='center')
    
    plt.title('Employment Outcomes by Industry for Recent Graduates')
    plt.xlabel('Year')
    plt.ylabel('Employment Count')
    plt.xticks(df['Year'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Use Streamlit to display the plot
    st.pyplot(plt)
def interactive_priority_ranking_heatmap(df):

    skill_categories = ["Specialized Industry Skills", "Business Skills", "Tech Skills", "Soft Skills", "Disruptive Tech Skills"]
    filtered_df = df[df['skill_group_category'].isin(skill_categories)]

    # Group by year, isic_section_name, industry_name, and skill_group_category and sum up the skill_group_rank
    grouped_sums = filtered_df.groupby(['year', 'isic_section_name', 'industry_name', 'skill_group_category'])['skill_group_rank'].sum().reset_index(name='sum_skill_group_rank')

    # Interactive Streamlit sidebar widgets
    industry_names = st.sidebar.multiselect('Select Industry', options=grouped_sums['industry_name'].unique(), default=grouped_sums['industry_name'].unique()[0])
    selected_year = st.sidebar.slider('Select Year', int(grouped_sums['year'].min()), int(grouped_sums['year'].max()), int(grouped_sums['year'].min()))

    # Filter based on selection
    specific_industry_grouped_sums = grouped_sums[(grouped_sums['industry_name'].isin(industry_names)) & (grouped_sums['year'] == selected_year)]

    # Calculate quantile-based rankings
    specific_industry_grouped_sums['quantile_rank'] = pd.qcut(specific_industry_grouped_sums['sum_skill_group_rank'], 5, labels=False) + 1
    specific_industry_grouped_sums['priority_rank'] = specific_industry_grouped_sums['quantile_rank'].apply(lambda x: 6 - x)

    # Pivot the DataFrame for heatmap visualization
    heatmap_data = specific_industry_grouped_sums.pivot(index='skill_group_category', columns='year', values='priority_rank')

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='inferno', fmt=".0f", linewidths=.5, cbar_kws={'label': 'Priority Rank (1=Highest, 5=Lowest)'})
    plt.title('Priority Ranking Heatmap of Skill Group Categories by Year')
    plt.ylabel('Skill Group Category')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the heatmap in Streamlit
    st.pyplot(plt)
def interactive_priority_ranking_group(df):
    """
    Displays a bar chart showing the distribution of skill group categories within a selected industry.
    """
    # Selectbox for industry selection
    industry_name = st.sidebar.selectbox('Select Industry', options=df['industry_name'].unique())
    
    # Filter dataframe by selected industry
    filtered_df = df[df['industry_name'] == industry_name]
    
    # Group by skill group category and count occurrences
    skill_group_distribution = filtered_df.groupby('skill_group_category').size().reset_index(name='Count')
    
    # Plot the distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Count', y='skill_group_category', data=skill_group_distribution, palette='coolwarm')
    plt.title(f'Skill Group Distribution in {industry_name}')
    plt.xlabel('Count')
    plt.ylabel('Skill Group Category')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

def interactive_top_skill_groups(df):
    """
    Displays the top N skill groups across all industries, based on user selection.
    """
    # Count occurrences of each skill group
    skill_group_counts = df['skill_group_name'].value_counts().reset_index(name='Count')
    skill_group_counts.columns = ['Skill Group', 'Count']
    
    # Slider for selecting the number of top skill groups to display
    num_skill_groups = st.sidebar.slider('Number of Top Skill Groups to Display', 1, len(skill_group_counts), 10)
    
    # Plot the top N skill groups
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='Skill Group', data=skill_group_counts.head(num_skill_groups), palette='viridis')
    plt.title(f'Top {num_skill_groups} Skill Groups Across Industries')
    plt.xlabel('Count')
    plt.ylabel('Skill Group')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)
def main():
    st.title("Worker Needs Dashboard")
    
    # Load data
    df1, df2, df34, df5, df6, df7, df8 = load_data()
    
    # Interactive selection for the dataset
    dataset_name = st.sidebar.selectbox("Select Dataset", (
        "Employment Projections", 
        "Occupational Employment and Wages", 
        "Employment Outcomes",
        "Top Skill Groups",
        "Skill Group Heatmap", 
        "Skill Group Distribution",
    ))
    
    
    if dataset_name == "Employment Projections":
        df1 = preprocess_df1(df1)
        visualization = st.sidebar.selectbox(
            "Select a Visualization", 
            [
                "Average Weekly Wages", 
                "Top 20 Industries by Net Growth", 
                "Top N Industries by Net Growth",
                "Average weekly N wage"
            ]
        )
        if visualization == "Average Weekly Wages":
            plot_average_weekly_wages(df1)
        elif visualization == "Top 20 Industries by Net Growth":
            plot_top_20_net_growth(df1)
        elif visualization == "Top N Industries by Net Growth":
            plot_top_n_net_growth_in(df1)
        elif visualization == "Average weekly N wage":
            interactive_avg_weekly_wages(df1)
    
    elif dataset_name == "Occupational Employment and Wages":
        show_bubble_chart(df2)
    
    elif dataset_name == "Employment Outcomes":
        interactive_employment_outcomes_by_industry(df6)
    
    elif dataset_name == "Skill Group Heatmap":
        interactive_priority_ranking_heatmap(df8)
    
    elif dataset_name == "Skill Group Distribution":
        interactive_priority_ranking_group(df8)
    elif dataset_name == "Top Skill Groups":
        interactive_top_skill_groups(df8)

if __name__ == "__main__":
    main()