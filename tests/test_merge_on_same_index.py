import pandas as pd
import numpy as np
from phenotypic.core.pipeline_parts._image_pipeline_core import ImagePipelineCore

# Create test dataframes with some identical columns
df1 = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6],
    'col3': [7, 8, 9]
}, index=pd.Index(['a', 'b', 'c'], name='test_index'))

df2 = pd.DataFrame({
    'col2': [4, 5, 6],  # Identical to df1.col2
    'col3': [10, 11, 12],  # Different values from df1.col3
    'col4': [13, 14, 15]
}, index=pd.Index(['a', 'b', 'c'], name='test_index'))

df3 = pd.DataFrame({
    'col5': [16, 17, 18],
    'col6': [19, 20, 21]
}, index=pd.Index(['a', 'b', 'c'], name='test_index'))

# Test the _merge_on_same_index function
result = ImagePipelineCore._merge_on_same_index([df1, df2, df3])

# Print the result
print("Result columns:", result.columns.tolist())
print("Expected columns: ['col1', 'col2', 'col3', 'col3_duplicate', 'col4', 'col5', 'col6']")

# Verify that duplicate column 'col2' was removed from df2 before merging
print("\nVerifying col2 values (should match df1):")
print(result['col2'])

# Verify that non-duplicate column 'col3' was kept from both dataframes
print("\nVerifying col3 values (should be from df1 since it's processed first):")
print(result['col3'])

# Create another test case with completely identical dataframes
df4 = pd.DataFrame({
    'colA': [1, 2, 3],
    'colB': [4, 5, 6]
}, index=pd.Index(['x', 'y', 'z'], name='another_index'))

df5 = pd.DataFrame({
    'colA': [1, 2, 3],
    'colB': [4, 5, 6]
}, index=pd.Index(['x', 'y', 'z'], name='another_index'))

# Test with identical dataframes
print("\nTesting with identical dataframes:")
result2 = ImagePipelineCore._merge_on_same_index([df4, df5])
print("Result2 columns:", result2.columns.tolist())
print("Expected columns: ['colA', 'colB']")
