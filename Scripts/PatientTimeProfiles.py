#Library import
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

#Class PatientTimeProfiles definition, initialization function to instantiate objects, receives 4 DataFrames
class PatientTimeProfiles:
    def __init__(self, study_population, prescriptions, treatments, diagnostics):
        self.IStudyPopulation = study_population
        self.PrescriptionsF = prescriptions
        self.Treatments = treatments
        self.Diagnostics = diagnostics

    #OUT OF USE
    #Create_sparse_matrix funciton definition
    def create_sparse_matrix(self):
        #Formatting of three date variables
        self.IStudyPopulation['PeriodStart'] = pd.to_datetime(self.IStudyPopulation['PeriodStart'], format="mixed")
        self.IStudyPopulation['PeriodEnd'] = pd.to_datetime(self.IStudyPopulation['PeriodEnd'], format="mixed")
        self.PrescriptionsF.loc[:,'Verordnungsdatum'] = pd.to_datetime(self.PrescriptionsF['Verordnungsdatum'],         format="mixed")

        # Sorted ATC_Codes
        atc_codes = sorted(self.PrescriptionsF['ATCX'].astype(str).unique())
       
        #Creates a sorted list of pid, PeriodNumber couples for row keys and a sorted list of atc_codes for column keys
        row_keys = sorted(set((row['PID'], row['PeriodNumber']) for _, row in self.IStudyPopulation.iterrows()))
        col_keys = sorted(atc_codes)

        #Creates mapping for both from numbered integers to the keys
        row_mapping = {key: idx for idx, key in enumerate(row_keys)}
        col_mapping = {code: idx for idx, code in enumerate(col_keys)}
   
        #Empty lists
        rows = []
        cols = []
        data = []
       
        # Dictionaries to store Age and Sex information for each row
        age_dict = {}
        sex_dict = {}

        # Iterate over each patient ID and corresponding period ranges
        for pid, group in self.IStudyPopulation.groupby('PID'):
            #Keeps prescriptions for that pid
            patient_prescriptions = self.PrescriptionsF[self.PrescriptionsF['PID'] == pid]
            #Iterates over the rows of the group with the different subperiods, saves the period start, end and number.
            for _, period_row in group.iterrows():
                period_start = period_row['PeriodStart']
                period_end = period_row['PeriodEnd']
                period_number = period_row["PeriodNumber"]
                age = period_row['Age']
                sex = period_row['Sex']
               
                # Boolean mask to find matching records for the specific subperiod in PrescriptionsF
                mask = (patient_prescriptions['Verordnungsdatum'] >= period_start) & (patient_prescriptions['Verordnungsdatum'] <= period_end)
                matching_records = patient_prescriptions[mask]
               
                row_idx = row_mapping[(pid, period_number)]
               
                age_dict[row_idx] = age
                sex_dict[row_idx] = sex
               
                # Add entries to sparse matrix lists, the index of the row, of the col, and a 1 to data
                for _, record in matching_records.iterrows():
                    atc_code = str(record['ATCX'])
                    if atc_code in col_mapping:
                       
                        col_idx = col_mapping[atc_code]
                        rows.append(row_idx)
                        cols.append(col_idx)
                        data.append(1)

        sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(row_keys), len(col_keys)))
                 
        #Prints mappings
        print("Row Mapping:", row_mapping)
        print("Column Mapping:", col_mapping)
        print("Age Dictionary:", age_dict)
        print("Sex Dictionary:", sex_dict)
       
        return sparse_matrix, age_dict, sex_dict
   
   
   
   
    def create_sparse_dataframe(self):
        # Formatting of date variables
        self.IStudyPopulation['PeriodStart'] = pd.to_datetime(self.IStudyPopulation['PeriodStart'], format="mixed")
        self.IStudyPopulation['PeriodEnd'] = pd.to_datetime(self.IStudyPopulation['PeriodEnd'], format="mixed")
        self.PrescriptionsF['Verordnungsdatum'] = pd.to_datetime(self.PrescriptionsF['Verordnungsdatum'], format="mixed")
        self.Treatments['Leistungsdatum'] = pd.to_datetime(self.Treatments['Leistungsdatum'], format="mixed")
        self.Diagnostics['Bezugsjahr'] = pd.to_datetime(self.Diagnostics['Bezugsjahr'], format="mixed")

        # Get all unique ATC codes to be used as columns
        atc_codes = sorted(self.PrescriptionsF['ATCX'].astype(str).unique())
       
        #Get all unique HEMI codes to be used as columns
        hemi_codes = sorted(self.Treatments['Leistung'].astype(str).unique(), key=int)
       
        #Get all unique ICD codes to be used as columns
        icd_codes = sorted(self.Diagnostics['ICD_Code'].astype(str).unique())
       
        # Initialize the DataFrame with the main columns
        result_data = {
            'PID': [],
            'Subperiod': [],
            'PeriodStart': [],
            'PeriodEnd': [],
            '1_Revision': [],
            'Age': [],
            'Sex': []
        }

        # Add a column for each ATC code with initial values as 0 (Initially a list for every ATC)
        for atc_code in atc_codes:
            result_data[f"ATCX_{atc_code}"] = []
           
        #Add a column for each HEMI code with initial values as 0
        for hemi_code in hemi_codes:
            result_data[f"HEMI_{hemi_code}"] = []
           
        # Add a column for each ICD code with initial values as 0
        for icd_code in icd_codes:
            result_data[f"AICDX_{icd_code}"] = []

        # Iterate over each patient ID and corresponding period ranges
        for pid, group in self.IStudyPopulation.groupby('PID'):
            # Keep prescriptions, treatments and diagnostics for that PID
            patient_prescriptions = self.PrescriptionsF[self.PrescriptionsF['PID'] == pid]
            patient_treatments = self.Treatments[self.Treatments['PID'] == pid]
            patient_diagnostics = self.Diagnostics[self.Diagnostics['PID'] == pid]

            # Iterate over the rows of the group with the different subperiods
            for _, period_row in group.iterrows():
                period_start = period_row['PeriodStart']
                period_end = period_row['PeriodEnd']
                period_number = period_row['PeriodNumber']
                age = period_row['Age']
                sex = period_row['Sex']
                revision = period_row['1_Revision']  # Default to 0 if '1_Revision' is not present

                # Boolean mask to find matching records for the specific subperiod in PrescriptionsF
                mask = (patient_prescriptions['Verordnungsdatum'] >= period_start) & (patient_prescriptions['Verordnungsdatum'] <= period_end)
                matching_records = patient_prescriptions[mask]
               
                #Boolean mask to find matching records for the specific subperiod in Treatments
                mask_two = (patient_treatments['Leistungsdatum'] >= period_start) & (patient_treatments['Leistungsdatum'] <= period_end)        
                matching_records_two = patient_treatments[mask_two]
               
                #Boolean mask to find matching records for the specific subperiod in Diagnostics
                mask_three = (patient_diagnostics['Bezugsjahr'] >= period_start) & (patient_diagnostics['Bezugsjahr'] <= period_end)
                matching_records_three = patient_diagnostics[mask_three]

                # Initialize a row of zeros for ATC codes
                atc_presence = {atc_code: 0 for atc_code in atc_codes}
               
                #Initialize a row of zeros for HEMI codes
                hemi_presence = {hemi_code: 0 for hemi_code in hemi_codes}
               
                #Initialize a row of zeros for ICD codes
                icd_presence = {icd_code: 0 for icd_code in icd_codes}

                # Mark `1` for each ATC code that was prescribed in this subperiod
                for atc_code in matching_records['ATCX'].unique():
                    if atc_code in atc_presence:
                        atc_presence[atc_code] = 1
                   
                # Mark `1` for each HEMI code that was prescribed in this subperiod
                for hemi_code in matching_records_two['Leistung'].unique():
                    hemi_code = str(hemi_code)
                    if hemi_code in hemi_presence:
                        hemi_presence[hemi_code] = 1
               
                #Mark '1' for each ICD code that was prescribed in this subperiod
                for icd_code in matching_records_three['ICD_Code'].unique():
                    icd_code = str(icd_code)
                    if icd_code in icd_presence:
                        icd_presence[icd_code] = 1

   
                # Append the details to the DataFrame's dictionary
                result_data['PID'].append(pid)
                result_data['Subperiod'].append(period_number)
                result_data['PeriodStart'].append(period_start)
                result_data['PeriodEnd'].append(period_end)
                result_data['1_Revision'].append(revision)
                result_data['Age'].append(age)
                result_data['Sex'].append(sex)
                       
                # Append ATC prescription status (1 or 0) for each ATC code
                for atc_code in atc_codes:
                    result_data[f"ATCX_{atc_code}"].append(atc_presence[atc_code])
                   
                # Append HEMI treatmetment status (1 or 0) for each HEMI code
                for hemi_code in hemi_codes:
                    result_data[f"HEMI_{hemi_code}"].append(hemi_presence[hemi_code])
                   
                #Append ICD diagnostic status (1 or 0) for each ICD code
                for icd_code in icd_codes:
                    result_data[f"AICDX_{icd_code}"].append(icd_presence[icd_code])
                   
                   
                   
        # Convert the dictionary to a DataFrame
        result_df = pd.DataFrame(result_data)
       
        return result_df
   
    #Similar but, does just for period following 30 days after operation date, multiplies the number of packages by the quantity and sum across all records within the study period for that particular medicine for that patient. Normalizes or puts in terms of "number of standard deviations away from the mean", so you get the mean quantity across patients and each patients quantity you substract that mean and divide over the sd of quantities across patients
    def create_sparse_dataframe_first_30_days(self):
        # Formatting date variables
        self.IStudyPopulation['PeriodStart'] = pd.to_datetime(self.IStudyPopulation['PeriodStart'], format="mixed")
        self.PrescriptionsF['Verordnungsdatum'] = pd.to_datetime(self.PrescriptionsF['Verordnungsdatum'], format="mixed")
        self.Treatments['Leistungsdatum'] = pd.to_datetime(self.Treatments['Leistungsdatum'], format="mixed")
        self.Diagnostics['Bezugsjahr'] = pd.to_datetime(self.Diagnostics['Bezugsjahr'], format="mixed")

        # Get unique codes for columns
        atc_codes = sorted(self.PrescriptionsF['ATCX'].astype(str).unique())
        hemi_codes = sorted(self.Treatments['Leistung'].astype(str).unique(), key=int)
        icd_codes = sorted(self.Diagnostics['ICD_Code'].astype(str).unique())

        # Initialize the result dictionary
        result_data = {
            'PID': [],
            'PeriodStart': [],
            'PeriodEnd': [],
            '1_Revision': [],
            'Age': [],
            'Sex': []
        }

        # Add columns for ATC, HEMI, and ICD codes
        for atc_code in atc_codes:
            result_data[f"ATCX_{atc_code}"] = []
        for hemi_code in hemi_codes:
            result_data[f"HEMI_{hemi_code}"] = []
        for icd_code in icd_codes:
            result_data[f"AICDX_{icd_code}"] = []

        # Process each PID
        for pid, group in self.IStudyPopulation.groupby('PID'):
            # Extract the first subperiod details
            first_subperiod = group.iloc[0]
            period_start = first_subperiod['PeriodStart']
            period_end = period_start + pd.Timedelta(days=30)
            age = first_subperiod['Age']
            sex = first_subperiod['Sex']
            revision = first_subperiod.get('1_Revision', 0)  # Default to 0 if not present

            # Filter prescriptions, treatments, and diagnostics for this PID and subperiod
            patient_prescriptions = self.PrescriptionsF[self.PrescriptionsF['PID'] == pid]
            matching_prescriptions = patient_prescriptions[
                (patient_prescriptions['Verordnungsdatum'] >= period_start) &
                (patient_prescriptions['Verordnungsdatum'] <= period_end)
            ]

            patient_treatments = self.Treatments[self.Treatments['PID'] == pid]
            matching_treatments = patient_treatments[
                (patient_treatments['Leistungsdatum'] >= period_start) &
                (patient_treatments['Leistungsdatum'] <= period_end)
            ]

            patient_diagnostics = self.Diagnostics[self.Diagnostics['PID'] == pid]
            matching_diagnostics = patient_diagnostics[
                (patient_diagnostics['Bezugsjahr'] >= period_start) &
                (patient_diagnostics['Bezugsjahr'] <= period_end)
            ]

            # Initialize presence dictionaries
            atc_presence = {atc_code: 0 for atc_code in atc_codes}
            hemi_presence = {hemi_code: 0 for hemi_code in hemi_codes}
            icd_presence = {icd_code: 0 for icd_code in icd_codes}

            # Calculate presence counts for ATC codes
            for atc_code in matching_prescriptions['ATCX'].unique():
                if atc_code in atc_presence:
                    total_quantity = matching_prescriptions.loc[
                        matching_prescriptions['ATCX'] == atc_code, 'Anzahl_Packungen'
                    ].fillna(0) * matching_prescriptions.loc[
                        matching_prescriptions['ATCX'] == atc_code, 'MENGE'
                    ].fillna(0)
                    atc_presence[atc_code] = total_quantity.sum()

            # Mark presence for HEMI codes
            for hemi_code in matching_treatments['Leistung'].unique():
                hemi_code = str(hemi_code)
                if hemi_code in hemi_presence:
                    hemi_presence[hemi_code] = 1

            # Mark presence for ICD codes
            for icd_code in matching_diagnostics['ICD_Code'].unique():
                icd_code = str(icd_code)
                if icd_code in icd_presence:
                    icd_presence[icd_code] = 1

            # Append data to the result dictionary
            result_data['PID'].append(pid)
            result_data['PeriodStart'].append(period_start)
            result_data['PeriodEnd'].append(period_end)
            result_data['1_Revision'].append(revision)
            result_data['Age'].append(age)
            result_data['Sex'].append(sex)

            # Append ATC, HEMI, and ICD presence data
            for atc_code in atc_codes:
                result_data[f"ATCX_{atc_code}"].append(atc_presence[atc_code])
            for hemi_code in hemi_codes:
                result_data[f"HEMI_{hemi_code}"].append(hemi_presence[hemi_code])
            for icd_code in icd_codes:
                result_data[f"AICDX_{icd_code}"].append(icd_presence[icd_code])

        # Convert the dictionary to a DataFrame
        result_df = pd.DataFrame(result_data)

        # Normalize the ATCX columns
        atcx_columns = [col for col in result_df.columns if col.startswith('ATCX_')]
        mean_std = result_df[atcx_columns].agg(['mean', 'std'])
        result_df[atcx_columns] = (result_df[atcx_columns] - mean_std.loc['mean']) / mean_std.loc['std']
       
       
        return result_df
   



    def create_sparse_dataframe_count(self):
        # Formatting of date variables
        self.IStudyPopulation['PeriodStart'] = pd.to_datetime(self.IStudyPopulation['PeriodStart'], format="mixed")
        self.IStudyPopulation['PeriodEnd'] = pd.to_datetime(self.IStudyPopulation['PeriodEnd'], format="mixed")
        self.PrescriptionsF['Verordnungsdatum'] = pd.to_datetime(self.PrescriptionsF['Verordnungsdatum'], format="mixed")
        self.Treatments['Leistungsdatum'] = pd.to_datetime(self.Treatments['Leistungsdatum'], format="mixed")
        self.Diagnostics['Bezugsjahr'] = pd.to_datetime(self.Diagnostics['Bezugsjahr'], format="mixed")

        # Get all unique ATC codes to be used as columns
        atc_codes = sorted(self.PrescriptionsF['ATCX'].astype(str).unique())
       
        #Get all unique HEMI codes to be used as columns
        hemi_codes = sorted(self.Treatments['Leistung'].astype(str).unique(), key=int)
       
        #Get all unique ICD codes to be used as columns
        icd_codes = sorted(self.Diagnostics['ICD_Code'].astype(str).unique())
       
        # Initialize the DataFrame with the main columns
        result_data = {
            'PID': [],
            'Subperiod': [],
            'PeriodStart': [],
            'PeriodEnd': [],
            '1_Revision': [],
            'Age': [],
            'Sex': []
        }

        # Add a column for each ATC code with initial values as 0 (Initially a list for every ATC)
        for atc_code in atc_codes:
            result_data[f"ATCX_{atc_code}"] = []
           
        #Add a column for each HEMI code with initial values as 0
        for hemi_code in hemi_codes:
            result_data[f"HEMI_{hemi_code}"] = []
           
        # Add a column for each ICD code with initial values as 0
        for icd_code in icd_codes:
            result_data[f"AICDX_{icd_code}"] = []

        # Iterate over each patient ID and corresponding period ranges
        for pid, group in self.IStudyPopulation.groupby('PID'):
            # Keep prescriptions, treatments and diagnostics for that PID
            patient_prescriptions = self.PrescriptionsF[self.PrescriptionsF['PID'] == pid]
            patient_treatments = self.Treatments[self.Treatments['PID'] == pid]
            patient_diagnostics = self.Diagnostics[self.Diagnostics['PID'] == pid]

            # Iterate over the rows of the group with the different subperiods
            for _, period_row in group.iterrows():
                period_start = period_row['PeriodStart']
                period_end = period_row['PeriodEnd']
                period_number = period_row['PeriodNumber']
                age = period_row['Age']
                sex = period_row['Sex']
                revision = period_row['1_Revision']  # Default to 0 if '1_Revision' is not present

                # Boolean mask to find matching records for the specific subperiod in PrescriptionsF
                mask = (patient_prescriptions['Verordnungsdatum'] >= period_start) & (patient_prescriptions['Verordnungsdatum'] <= period_end)
                matching_records = patient_prescriptions[mask]
               
                #Boolean mask to find matching records for the specific subperiod in Treatments
                mask_two = (patient_treatments['Leistungsdatum'] >= period_start) & (patient_treatments['Leistungsdatum'] <= period_end)        
                matching_records_two = patient_treatments[mask_two]
               
                #Boolean mask to find matching records for the specific subperiod in Diagnostics
                mask_three = (patient_diagnostics['Bezugsjahr'] >= period_start) & (patient_diagnostics['Bezugsjahr'] <= period_end)
                matching_records_three = patient_diagnostics[mask_three]

                # Initialize a row of zeros for ATC codes
                atc_presence = {atc_code: 0 for atc_code in atc_codes}
               
                #Initialize a row of zeros for HEMI codes
                hemi_presence = {hemi_code: 0 for hemi_code in hemi_codes}
               
                #Initialize a row of zeros for ICD codes
                icd_presence = {icd_code: 0 for icd_code in icd_codes}

                # Mark 1 for each ATC code that was prescribed in this subperiod
                for atc_code in matching_records['ATCX'].unique():
                    if atc_code in atc_presence:
                        # Calculate the product for rows with the current atc_code
                        total_quantity = matching_records.loc[
                            matching_records['ATCX'] == atc_code, 'Anzahl_Packungen'
                        ].fillna(0) * matching_records.loc[
                            matching_records['ATCX'] == atc_code, 'MENGE'
                        ].fillna(0)
                        # Sum the product and assign to atc_presence
                        atc_presence[atc_code] = total_quantity.sum()
                   
                # Mark 1 for each HEMI code that was prescribed in this subperiod
                for hemi_code in matching_records_two['Leistung'].unique():
                    hemi_code = str(hemi_code)
                    if hemi_code in hemi_presence:
                        hemi_presence[hemi_code] += 1
               
                #Mark '1' for each ICD code that was prescribed in this subperiod
                for icd_code in matching_records_three['ICD_Code'].unique():
                    icd_code = str(icd_code)
                    if icd_code in icd_presence:
                        icd_presence[icd_code] += 1

   
                # Append the details to the DataFrame's dictionary
                result_data['PID'].append(pid)
                result_data['Subperiod'].append(period_number)
                result_data['PeriodStart'].append(period_start)
                result_data['PeriodEnd'].append(period_end)
                result_data['1_Revision'].append(revision)
                result_data['Age'].append(age)
                result_data['Sex'].append(sex)
                       
                # Append ATC prescription status (1 or 0) for each ATC code
                for atc_code in atc_codes:
                    result_data[f"ATCX_{atc_code}"].append(atc_presence[atc_code])
                   
                # Append HEMI treatmetment status (1 or 0) for each HEMI code
                for hemi_code in hemi_codes:
                    result_data[f"HEMI_{hemi_code}"].append(hemi_presence[hemi_code])
                   
                #Append ICD diagnostic status (1 or 0) for each ICD code
                for icd_code in icd_codes:
                    result_data[f"AICDX_{icd_code}"].append(icd_presence[icd_code])
                   
                   
                   
        # Convert the dictionary to a DataFrame
        result_df = pd.DataFrame(result_data)
       
        # Normalize the ATCX columns
        atcx_columns = [col for col in result_df.columns if col.startswith('ATCX_')]
        mean_std = result_df[atcx_columns].agg(['mean', 'std'])
        result_df[atcx_columns] = (result_df[atcx_columns] - mean_std.loc['mean']) / mean_std.loc['std']
       
        # Normalize HEMI columns
        if hemi_presence:
            hemi_values = np.array(list(hemi_presence.values()))
            hemi_mean = hemi_values.mean()
            hemi_std = hemi_values.std() if hemi_values.std() > 0 else 1  # Prevent division by zero
            normalized_hemi = {k: (v - hemi_mean) / hemi_std for k, v in hemi_presence.items()}
            hemi_presence.update(normalized_hemi)  # Update with normalized values

# Normalize ICD columns
        if icd_presence:
            icd_values = np.array(list(icd_presence.values()))
            icd_mean = icd_values.mean()
            icd_std = icd_values.std() if icd_values.std() > 0 else 1  # Prevent division by zero
            normalized_icd = {k: (v - icd_mean) / icd_std for k, v in icd_presence.items()}
            icd_presence.update(normalized_icd)  # Update with normalized values


        return result_df
   
   
   
   
   
    #OUT OF USE
    def create_dataframe(self):
        # Formatting of date variables
        self.IStudyPopulation['PeriodStart'] = pd.to_datetime(self.IStudyPopulation['PeriodStart'], format="mixed")
        self.IStudyPopulation['PeriodEnd'] = pd.to_datetime(self.IStudyPopulation['PeriodEnd'], format="mixed")
        self.PrescriptionsF['Verordnungsdatum'] = pd.to_datetime(self.PrescriptionsF['Verordnungsdatum'], format="mixed")
        #self.Treatments['Leistungsdatum'] = pd.to_datetime(self.Treatments['Leistungsdatum'], format="mixed")

        # Get all unique ATC codes to be used as columns
        atc_codes = sorted(self.PrescriptionsF['ATCX'].astype(str).unique())
   
        # Get all unique HEMI codes to be used as columns
        #hemi_codes = sorted(self.Treatments['Leistung'].astype(str).unique(), key=int)
   
       
        rows = []
        # Iterate over each patient ID and corresponding period ranges
        for pid, group in self.IStudyPopulation.groupby('PID'):
            # Keep prescriptions for that PID
            patient_prescriptions = self.PrescriptionsF[self.PrescriptionsF['PID'] == pid]
            #patient_treatments = self.Treatments[self.Treatments['PID'] == pid]

            # Iterate over the rows of the group with the different subperiods
            for _, period_row in group.iterrows():
                period_start = period_row['PeriodStart']
                period_end = period_row['PeriodEnd']
                period_number = period_row['PeriodNumber']
                age = period_row['Age']
                sex = period_row['Sex']
                revision = period_row['1_Revision']

                # Boolean mask to find matching records for the specific subperiod in PrescriptionsF
                mask = (patient_prescriptions['Verordnungsdatum'] >= period_start) & (patient_prescriptions['Verordnungsdatum'] <= period_end)
                matching_records = patient_prescriptions[mask]
           
                # Boolean mask to find matching records for the specific subperiod in Treatments
                #mask_two = (patient_treatments['Leistungsdatum'] >= period_start) & (patient_treatments['Leistungsdatum'] <= period_end)
                #matching_records_two = patient_treatments[mask_two]

                # Initialize a row of zeros for ATC codes
                atc_presence = {atc_code: 0 for atc_code in atc_codes}
           
                # Initialize a row of zeros for HEMI codes
                #hemi_presence = {hemi_code: 0 for hemi_code in hemi_codes}

                # Mark `1` for each ATC code that was prescribed in this subperiod
                for atc_code in matching_records['ATCX'].unique():
                    if atc_code in atc_presence:
                        atc_presence[atc_code] = 1

                # Mark `1` for each HEMI code that was prescribed in this subperiod
                #for hemi_code in matching_records_two['Leistung'].unique():
                    #if hemi_code in hemi_presence:
                        #hemi_presence[hemi_code] = 1

                # Combine all information into a single row dictionary
                row = {
                    'PID': pid,
                    'Subperiod': period_number,
                    '1_Revision': revision,
                    'Age': age,
                    'Sex': sex,
                    **{f"ATCX_{atc_code}": atc_presence[atc_code] for atc_code in atc_codes},
                    #**{f"HEMI_{hemi_code}": hemi_presence[hemi_code] for hemi_code in hemi_codes}
                }

                # Append the row to the list
                rows.append(row)

        # Convert the list of rows into a DataFrame
        result_df = pd.DataFrame(rows)

        # Return the resulting DataFrame
        return result_df
