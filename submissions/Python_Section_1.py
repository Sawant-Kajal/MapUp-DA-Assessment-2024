from typing import Dict, List, Any
import pandas as pd
import re, polyline
import numpy as np
import os


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        end = min(i + n, length)
        for j in range(end - 1, i - 1, -1):
            result.append(lst[j])
    
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    sorted_length_dict = dict(sorted(length_dict.items()))
    
    return sorted_length_dict


def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    
    items = {}
    
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{new_key}[{index}]", sep=sep))
                else:
                    items[f"{new_key}[{index}]"] = item
        else:
            items[new_key] = value
            
    return items


def unique_permutations(nums: List[int]) -> List[List[int]]:
    
    def backtrack(path: List[int], used: List[bool]):
        if len(path) == len(nums):
            result.append(path[:])  # Append a copy of the current path
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue
            
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()  # Backtrack
            used[i] = False

    nums.sort()
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    return result


def find_all_dates(text: str) -> List[str]:
    
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    
    combined_pattern = '|'.join(patterns)
    matches = re.findall(combined_pattern, text)
    
    return matches


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    
    R = 6371000  # Radius of the Earth in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance in meters
    return distance

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]  # First distance is 0
    for i in range(1, len(df)):
        dist = haversine(df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'],
                         df.loc[i, 'latitude'], df.loc[i, 'longitude'])
        distances.append(dist)

    df['distance'] = distances
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    
    n = len(matrix)
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    
    day_map = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    base_date = pd.to_datetime('2023-01-02')
    df['start_date'] = df['startDay'].map(day_map).apply(lambda x: base_date + pd.Timedelta(days=x))
    df['end_date'] = df['endDay'].map(day_map).apply(lambda x: base_date + pd.Timedelta(days=x))
    
    df['start'] = pd.to_datetime(df['start_date'].dt.date.astype(str) + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['end_date'].dt.date.astype(str) + ' ' + df['endTime'])

    df.set_index(['id', 'id_2'], inplace=True)

    grouped = df.groupby(level=['id', 'id_2'])
    results = {}

    for name, group in grouped:
        full_day = (group['end'].max() - group['start'].min()) >= pd.Timedelta(hours=24)
        days_present = group['start'].dt.dayofweek.unique()
        full_week = set(days_present) == set(range(7))

        results[name] = not (full_day and full_week)

    boolean_series = pd.Series(results, index=pd.MultiIndex.from_tuples(results.keys(), names=['id', 'id_2']))

    return boolean_series

print("----------------------------Question 1: Reverse List by N Elements----------------------------------")
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]



print("----------------------------Question 2: Lists & Dictionaries----------------------------------")
print(group_by_length(["one", "two", "three", "four"]))



print("----------------------------Question 3: Flatten a Nested Dictionary----------------------------------")
print(flatten_dict(nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
)
)


print("----------------------------Question 4: Generate Unique Permutations----------------------------------")
print(unique_permutations([1,1,2]))



print("----------------------------Question 5: Find All Dates in a Text----------------------------------")
print(find_all_dates("I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."))



print("----------------------------Question 6: Decode Polyline, Convert to DataFrame with Distances----------------------------------")
print(polyline_to_dataframe("u{~vFvyys@fS]"))


print("----------------------------Question 7: Matrix Rotation and Transformation----------------------------------")

print(rotate_and_multiply_matrix( [[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


print("----------------------------Question 8: Time Check----------------------------------")


folder_path= os.getcwd()  
path = os.path.join(folder_path,"datasets")
file_path = os.path.join(path,"dataset-1.csv")
print(file_path)
data_path = pd.read_csv(file_path)
df = time_check(data_path)
print(df)


