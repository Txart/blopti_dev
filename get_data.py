import requests
import pandas as pd
import sys
from pathlib import Path
if sys.version_info[0] < 3:  # Fork for python2 and python3 compatibility
    from StringIO import StringIO
else:
    from io import StringIO
import datetime

from estimate_daily_et import compute_ET

#%%
def get_instantaneous_weather_data():
    DID =  '001D0A00F65E' # DEVIDE ID
    password = 'putri061112'
    apiToken = 'CF4BCA6F352A46358FAF00E7E9009516'
    api_url = 'https://api.weatherlink.com/v1/NoaaExt.json?user=' + DID + '&pass=' + password + '&apiToken=' + apiToken

    
    # This is where the request happens
    response = requests.request(method='POST', url=api_url)
    print('HTTP response status code: ', response.status_code)
    data = response.content.decode('utf-8')
    return pd.read_json(data)
 

# df = get_wtd_data(measured_quantity='water_table_1', output_mode='json', n_records=100)
def get_day_rainfall(): # WARNING: I DO NOT KNOW WHETHER THIS IS ACTUALLY DAILY RAINFALL OR NOT
    df_weather = get_instantaneous_weather_data()
    return float(df_weather['davis_current_observation']['rain_day_in'])


def read_historic_weather_data(fn_w):   
    # Read P and ET from weather station data.
    df_w = pd.read_excel(fn_w, skiprows=6) # thousands reads comma as dot!
    df_w[['Date','Time', 'Meridiam']] = df_w['Date & Time'].str.split(" ",expand=True,) # split date and time into 2 columns
    
    return df_w

# TODO: standarize time and date input
def get_wt_data():
    url_base = "https://dashboard.sustainabilitytech.com/api/"
    api_key = '61b8b35c1a42075c2f2de046ffe1975c908bc41355bda38f15efcf9cb9e7/'
    version = 'v1/'
    format_csv = 'csv/'
    collection = 'measurement/module[waterTable]/'
    created = 'created=-1/100000///'
    #join = '>10-03-2020&<12-03-2020'
    sensor = 'sensor=devId:label'
    
    url = url_base + api_key + version + format_csv + collection + created + sensor
    
    response = requests.request(method='POST', url=url)
    data = response.text
    strdata = StringIO(data)
    return pd.read_csv(strdata)

def compute_ET_dataframe(row):
    ET, _, _, _, _ = compute_ET(row['julian_day'], row['T_max'], row['T_ave'],
                      row['T_min'], row['rel_hum'], row['windspeed'], row['air_pressure'])
    return ET


def string_to_datetime(date_string):
    # formats string dates as datetime objects
    # date_string should be inputed as:
    # 'dd/mm/yy' (weather station) or as
    # 'yyyy-mm-dd' (wtdsensors)
    if '/' in date_string: # weather station format
        d, m, y = [int(elem) for elem in date_string.split('/')]
        y = int(y) + 2000 # add 2000 years. Otherwise, it thinks it's year 18 AC
        return datetime.date(year=y, month=m, day=d)
    
    elif '-' in date_string: # wtd sensor format
        y, m, d = [int(elem) for elem in date_string.split('-')]
        return datetime.date(year=y, month=m, day=d)
    
    else:
        raise ValueError('the date format is not understood')
        
    
def datetime_to_julian(date_string):
    # date string should be of the format 'dd/mm/yy'
    dt = string_to_datetime(date_string)
    if dt.strftime('%y')=='18':
        add_year = 0 # data starts in 2018
    elif dt.strftime('%y')=='19':
        add_year = 1
    elif dt.strftime('%y')=='20':
        add_year = 2
    elif dt.strftime('%y')=='17':
        add_year = -1    
    else:
        raise ValueError('Year value outside scope')
        
    return int(dt.strftime('%j')) + 365*add_year

def clean_WTD_data(df):
    df = df.loc[:,'convertedValue': 'sensor-label'] # slice relevant info
    new = df['created'].str.split('T', n=1, expand=True)
    time = new[1].str.slice(stop=8)
    df['date']=new[0]
    df['time']=time
    df.drop(columns=['created'], inplace=True)
    
    return df

def clean_weather_data(df):
    df = df[df['Temp - °C'] != '--'] # drop missing Temperature values
    df = df[df['Hum - %'] != '--'] # drop also missing relative humidity values. This should be enough
    
    df['T'] = df['Temp - °C'].astype(float)
    df['rel_hum'] = df['Hum - %'].astype(float, errors='ignore')
    df['windspeed'] = df['Wind Speed - km/h'].astype(float)
    df['Barometer - mb'] = df['Barometer - mb'].astype(float)
    
    return df

def aggregate_wtd_to_daily(wt_df):
    #   - First, separate by sensor
    #   - Then, aggregate daily
    sensor_names = list(wt_df['sensor-label'].dropna().unique())
    daily_wtd_df = pd.DataFrame(columns=['date', 'julian_day', 'WTD','sensor_label'])
    
    for sname in sensor_names:
        df = wt_df[wt_df['sensor-label']==sname]
        df_daily = pd.DataFrame(columns=['date', 'julian_day', 'WTD'])
        df_daily['WTD'] = df.groupby('date', sort=False)['convertedValue'].mean().apply(lambda x: x/100) # from cm to m
        df_daily['date'] = df_daily.index
        df_daily['sensor_label'] = sname
        df_daily['julian_day'] = df_daily['date'].apply(datetime_to_julian)
        
        daily_wtd_df = daily_wtd_df.append(df_daily)
        
    return daily_wtd_df

def aggregate_weather_to_daily(weather_df):

    daily_weather_df = pd.DataFrame(columns=['Date','julian_day', 'T_ave', 'T_min', 'T_max', 'P', 'ET', 'rel_hum', 'windspeed'])
    
    daily_weather_df['T_ave'] = weather_df.groupby('Date', sort=False)['T'].mean()
    daily_weather_df['Date'] = daily_weather_df.index
    daily_weather_df['T_max'] = weather_df.groupby('Date', sort=False)['T'].max()
    daily_weather_df['T_min'] = weather_df.groupby('Date', sort=False)['T'].min()
    daily_weather_df['P'] = weather_df.groupby('Date', sort=False)['Rain - mm'].sum()
    daily_weather_df['julian_day'] = daily_weather_df['Date'].apply(datetime_to_julian)
    daily_weather_df['rel_hum'] = weather_df.groupby('Date', sort=False)['rel_hum'].mean()
    daily_weather_df['windspeed'] = weather_df.groupby('Date', sort=False)['windspeed'].mean()
    daily_weather_df['windspeed'] = daily_weather_df['windspeed'].apply(lambda x: x*1000/3600) # from km/h to m/s
    daily_weather_df['air_pressure'] = weather_df.groupby('Date', sort=False)['Barometer - mb'].mean().apply(lambda x: x/10) # mbar to kPa
    
    daily_weather_df.index = daily_weather_df['julian_day']
    return daily_weather_df

def compute_and_append_ET(daily_weather_df):
    daily_weather_df['ET'] = daily_weather_df.apply(compute_ET_dataframe, axis=1)
    return daily_weather_df

def wtd_dictionary_of_dataframes(wtd_df):
    wtd_df.index = wtd_df['julian_day']
    
    sensor_names = list(wtd_df['sensor_label'].dropna().unique())
    dfs_by_sensor_label = {} # dictionary of dataframes: {sensor name : dataframe}
    for sname in sensor_names:
        if 'wi' in sname: 
            continue # leave out wi-dosan-xx sensors
        if 'DOSAN' in sname: # have to change DOSAN 1a into DOSAN 01a etc.
            numberletter = sname.split(' ')[1]
            try: # numbers without letters -> canal sensors
                number = int(numberletter)
                letter = 'X'
            except: # if cannot convert to int, then there's a letter! 
                number, letter = (int(numberletter[:-1]), numberletter[-1])
            number = f'{number:02d}' # add leading zeros
            numberletter = number + letter
            
            modified_sname = 'DOSAN' + ' ' + numberletter
        elif 'DAYUN' in sname: # change DAYUN 01 to DAYUN 01X for sensor canals
            numberletter = sname.split(' ')[1]
            try: # numbers without letters -> canal sensors
                number = int(numberletter)
                letter = 'X'
                number = f'{number:02d}' # add leading zeros
                numberletter = number + letter
            except: # no changes if not in canals
                numberletter = numberletter
            
            modified_sname = 'DAYUN' + ' ' + numberletter
        else:
            modified_sname = sname # no changes in other sensor names
            
        df_sen = wtd_df[wtd_df['sensor_label'] == sname]

        dfs_by_sensor_label[modified_sname] = df_sen
        
    return dfs_by_sensor_label

def strings_containing_substring(string_list, substring):
    return [name for name in string_list if substring in name]

def transects_and_weather_together(dfs_by_sensor_label, daily_weather_df):
    sensor_names = list(dfs_by_sensor_label.keys())
    sensor_identifiers = ['P', 'DAYUN', 'DOSAN']
    dfs_by_transects = {}
    
    for si in sensor_identifiers:
        sn_subset = strings_containing_substring(sensor_names, si) # look for sensors with si in their name
    
        for n in range(1,50): # sensor name numbers are below 50
            if si == 'P':
                sensor_number = str(n).zfill(3)
                
            elif si == 'DAYUN' or si == 'DOSAN':
                sensor_number = str(n).zfill(2)
                
            else:
                raise ValueError('sensor name could not be identified')
    
            sn_sub_subset = strings_containing_substring(sn_subset, sensor_number) # inside sensor names, separate by number and type
            if not sn_sub_subset == []:
                dataframe_list = []
                for name in sn_sub_subset:
                    df = dfs_by_sensor_label[name]
                    # names for columns
                    if 'X' in name: # all columns labelled with an X are canal columns
                        new_name_col = 'sensor_0'
                    elif si == 'P':
                            new_name_col = 'sensor_1' # P sensors have only 1 sensor apart from the canal one
                    elif si == 'DAYUN' or si == 'DOSAN':
                        if 'a' in name:
                            new_name_col = 'sensor_1'
                        elif 'b' in name:
                            new_name_col = 'sensor_2'
                        elif 'c' in name:
                            new_name_col = 'sensor_3'
                        elif 'd' in name:
                            new_name_col = 'sensor_4'
                        elif 'e' in name:
                            new_name_col = 'sensor_5'
                        
                    df.rename(columns={'WTD': new_name_col}, inplace=True)
                    if 'date' in df.columns:
                        df.drop(columns=['date', 'sensor_label'], inplace=True)
                    dataframe_list.append(df)
                # concatenate the 2 df only when they have a common julian day
                dataframe_list.append(daily_weather_df)    
                df = pd.concat(dataframe_list, axis=1, join='inner')
                df = df.loc[:,~df.columns.duplicated()] # remove duplicated columns (julian_day)
                
                df['julian_day'] = df['julian_day'].astype(int)
                df = df.sort_index() # sort by julian day
                
                dfs_by_transects[si + sensor_number] = df
    
    return dfs_by_transects
   
#%%
    
def main(fn_weather_data):
    """
     Get data
    """
    # WTD
    wt_df = get_wt_data() 
    
    # Weather
    fn_weather_data = str(fn_weather_data.absolute())
    weather_df = read_historic_weather_data(fn_weather_data)
    
    """
     Clean data
    """
    wt_df = clean_WTD_data(wt_df)
    weather_df = clean_weather_data(weather_df)
    
    """
     Compute ET.
     Convert data to daily
    """
    daily_wtd_df = aggregate_wtd_to_daily(wt_df)
    daily_weather_df = aggregate_weather_to_daily(weather_df)
    daily_weather_df = compute_and_append_ET(daily_weather_df)
    
    """
     Organize sensor data into dictionary
    """
    # Dictionary of dataframes, keys are sensor names
    dfs_by_sensor_label = wtd_dictionary_of_dataframes(daily_wtd_df)
    
    """
     Put transects together and append weather data
    """       
    dfs_by_transects = transects_and_weather_together(dfs_by_sensor_label, daily_weather_df)

    return dfs_by_transects           
                                     
            


    








