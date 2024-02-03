import csv

csv_file_path = '5 20-21 CTE Demographics .xlsx - EnrollmentReport.csv'
output_file_path = 'output.csv'

# Open the CSV file in read mode
with open(csv_file_path, 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Create a list to store rows that need to be kept
    rows_to_keep = []

    for row in csv_reader:
        # Check if the 5th to 22nd values are not all 'n/a'
        if not all(value == 'n/a' for value in row[4:21]):
            # If at least one value is not 'n/a', add the row to the list
            rows_to_keep.append(row)

# Open the CSV file in write mode to create a new file
with open(output_file_path, 'w', newline='') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)

    # Write the rows to the new CSV file
    csv_writer.writerows(rows_to_keep)

print("Rows with 'n/a' in 5th-22nd values removed.")

# Read the 5th row

    # Print or process the data in the 5th row


