import os
import pandas as pd

shortname_to_filename = {
    'alldata': r'arc_metadata.csv',
    'clean_apx': r'arc_metadata - totally clean APX.csv',
    'slightly_dirty_apx': r'arc_metadata - slightly dirty APX.csv',
}


def get_df(shortname=r'alldata'):
    path = r'C:\dev\code\machinelearning\data'
    file = shortname_to_filename[shortname]
    filepath = os.path.join(path, file)
    df = pd.read_csv(filepath)
    return df


select_columns = {
    'CurveID': False,  # exclude
    'CurveName': False,  # exclude
    'DataSource': True,
    'Commodity': True,
    'ProductOrMarket': True,
    'Location': True,
    'ContractType': True,
    'PriceType': True,
    'Granularity': True,
    'PublicationFrequency': True,
    'PublicationCalendar': False,  # exclude <-- not populated
    'Currency': True,
    'CurveUnit': True,
    'CurveType': True,
    'UsageType': True,
    'Description': False,  # exclude
    'OwnerRole': True,
    'Destination': True,
    'CurveStatus': True,
    'SettlementTime': False,  # exclude <-- not populated
    'ProductCode': False,  # exclude <-- not populated
    'CreatedBy': False,  # exclude
    'CreatedDateTimeUtc': False,  # exclude
    'GMACurveID': False,  # exclude
    'TimeOfDay': True,
    'Timezone': False,  # exclude <-- not populated
    'SurfaceContractType': False,  # exclude <-- not populated
    'SurfaceStrikeType': False,  # exclude <-- not populated
}
# only if value is True
select_columns = [key if value else None for (key, value) in
                  select_columns.items()]
# remove None
select_columns = [item for item in select_columns if item]
