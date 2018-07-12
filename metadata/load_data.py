import os
import pandas as pd


def get_df():
    file = r'arc_metadata.csv'
    path = r'C:\dev\code\machinelearning\data'
    filepath = os.path.join(path, file)
    df = pd.read_csv(filepath)
    return df

select_columns = {
    'CurveID': False,               # exclude
    'CurveName': False,             # exclude
    'DataSource': True,
    'Commodity': True,
    'ProductOrMarket': True,
    'Location': True,
    'ContractType': True,
    'PriceType': True,
    'Granularity': True,
    'PublicationFrequency': True,
    'PublicationCalendar': True,
    'Currency': True,
    'CurveUnit': True,
    'CurveType': True,
    'UsageType': True,
    'Description': False,           # exclude
    'OwnerRole': True,
    'Destination': True,
    'CurveStatus': True,
    'SettlementTime': True,
    'ProductCode': True,
    'CreatedBy': False,             # exclude
    'CreatedDateTimeUtc': False,    # exclude
    'GMACurveID': False,            # exclude
    'TimeOfDay': True,
    'Timezone': True,
    'SurfaceContractType': True,
    'SurfaceStrikeType': True,
    }
# only if value is True
select_columns = [key if value else None for (key, value) in select_columns.items()]
# remove None
select_columns = [item for item in select_columns if item]