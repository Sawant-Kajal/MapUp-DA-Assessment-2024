import os
import pandas as pd
import numpy as np
from datetime import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_distance_matrix(df):
    

    toll_locations = sorted(set(df['id_start']).union(df['id_end']))

    distance_matrix = pd.DataFrame(
        data=np.inf,  
        index=toll_locations,
        columns=toll_locations
    )

    np.fill_diagonal(distance_matrix.values, 0)

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        distance_matrix.loc[id_start, id_end] = distance
        distance_matrix.loc[id_end, id_start] = distance  
    for k in toll_locations:
        for i in toll_locations:
            for j in toll_locations:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix

def unroll_distance_matrix(distance_df: pd.DataFrame) -> pd.DataFrame:
    unrolled_data = []

    for id_start in distance_df.index:
        for id_end in distance_df.columns:
            if id_start != id_end:  
                distance = distance_df.loc[id_start, id_end]
                unrolled_data.append((id_start, id_end, distance))
    
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    unrolled_df.reset_index(drop=True, inplace=True)

    return unrolled_df




def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Calculate the average distance for the reference_id
    average_distance = df[df['id_start'] == reference_id]['distance'].mean()

    if np.isnan(average_distance):  # Check if average_distance is NaN
        return []  # Return an empty list if the reference_id is not found

    # Calculate the threshold values
    lower_threshold = average_distance * 0.9  # 10% below
    upper_threshold = average_distance * 1.1  # 10% above

    # Find all id_start values within the threshold
    ids_within_threshold = df[
        (df['id_start'] != reference_id) &  # Exclude the reference ID itself
        (df['distance'] >= lower_threshold) &
        (df['distance'] <= upper_threshold)
    ]['id_start'].unique()

    # Convert to standard Python integers and return sorted list
    return sorted(map(int, ids_within_threshold))

def calculate_toll_rate(unrolled_df: pd.DataFrame) -> pd.DataFrame:
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rate_coefficients.items():
        unrolled_df[vehicle] = unrolled_df['distance'] * rate

    return unrolled_df

def calculate_time_based_toll_rates(toll_rates_df: pd.DataFrame) -> pd.DataFrame:
    discount_factors = {
        'weekday': {
            (time(0, 0), time(10, 0)): 0.8,
            (time(10, 0), time(18, 0)): 1.2,
            (time(18, 0), time(23, 59)): 0.8,
        },
        'weekend': 0.7
    }

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    results = []

    for (id_start, id_end), group in toll_rates_df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            for hour in range(24):
                start_time = time(hour, 0)
                vehicle_rates = {}
                
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    original_rate = group[vehicle].values[0]
                    
                    if day in ["Saturday", "Sunday"]:
                        discount = discount_factors['weekend']
                        vehicle_rates[vehicle] = original_rate * discount
                    else:
                        for time_range, discount in discount_factors['weekday'].items():
                            if time_range[0] <= start_time < time_range[1]:
                                vehicle_rates[vehicle] = original_rate * discount
                                break

                results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': time(hour, 59),  # End time is 59 minutes later
                    **vehicle_rates
                })

    final_df = pd.DataFrame(results)
    return final_df

if __name__ == "__main__":
    folder_path= os.getcwd()  
    path = os.path.join(folder_path,"datasets")
    file_path = os.path.join(path,"dataset-2.csv")
    print(file_path)
    data_path = pd.read_csv(file_path)
    reference_id = 1004356 
    
    # data_path = os.path.join(current_dir, 'datasets', 'dataset-2.csv')
    
    print("----------------------------QUESTION 9 START------------------------------")
    distance_matrix = calculate_distance_matrix(data_path)
    print(distance_matrix)
    print("@"*25)
    print("----------------------------QUESTION 9 DONE------------------------------")

    
    print("----------------------------QUESTION 10 START------------------------------")
    unrolled_df = unroll_distance_matrix(distance_matrix)
    print(unrolled_df)
    print("@"*25)
    print("----------------------------QUESTION 10 DONE------------------------------")

    
    print("----------------------------QUESTION 11 START------------------------------")
    ten_percentage_threshold = find_ids_within_ten_percentage_threshold(unrolled_df,reference_id)
    print(f"{reference_id}: {ten_percentage_threshold}")
    print("@"*25)
    print("----------------------------QUESTION 11 DONE------------------------------")

    
    print("----------------------------QUESTION 12 START------------------------------")
    toll_rates_df = calculate_toll_rate(unrolled_df)
    print(toll_rates_df)
    print("@"*25)
    print("----------------------------QUESTION 12 DONE------------------------------")

    
    print("----------------------------QUESTION 13 START------------------------------")
    time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rates_df)
    print(time_based_toll_rates_df)
    print("@"*25)
    print("----------------------------QUESTION 13 DONE------------------------------")
