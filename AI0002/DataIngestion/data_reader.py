#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This file is automatically generated by AION for AI0002_1 usecase.
File generation time: 2023-04-13 14:41:26
'''
#Standard Library modules
import sqlite3

#Third Party modules
import pandas as pd 
from pathlib import Path
from influxdb import InfluxDBClient

def dataReader(reader_type, target_path=None, config=None):
    if reader_type == 'sqlite':
        return sqlite_writer(target_path=target_path)
    elif reader_type == 'influx':
        return Influx_writer(config=config)
    else:
        raise ValueError("'{reader_type}' not added during code generation")

class sqlite_writer():
    def __init__(self, target_path):
        self.target_path = Path(target_path)
        database_file = self.target_path.stem + '.db'
        self.db = sqlite_db(self.target_path, database_file)

    def file_exists(self, file):
        if file:
            return self.db.table_exists(file)
        else:
            return False

    def read(self, file):
        return self.db.read(file)
        
    def write(self, data, file):
        self.db.write(data, file)

    def close(self):
        self.db.close()

class sqlite_db():

    def __init__(self, location, database_file=None):
        if not isinstance(location, Path):
            location = Path(location)
        if database_file:
            self.database_name = database_file
        else:
            self.database_name = location.stem + '.db'
        db_file = str(location/self.database_name)
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.tables = []

    def table_exists(self, name):
        if name in self.tables:
            return True
        elif name:
            query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}';"
            listOfTables = self.cursor.execute(query).fetchall()
            if len(listOfTables) > 0 :
                self.tables.append(name)
                return True
        return False

    def read(self, table_name):
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)

    def create_table(self,name, columns, dtypes):
        query = f'CREATE TABLE IF NOT EXISTS {name} ('
        
        for column, data_type in zip(columns, dtypes):
            query += f"'{column}'   TEXT,"
        query = query[:-1]
        query += ');'
        self.conn.execute(query)
        return True
        
    def write(self,data, table_name):
        if not self.table_exists(table_name):
            self.create_table(table_name, data.columns, data.dtypes)
        tuple_data = list(data.itertuples(index=False, name=None))
        insert_query = f'INSERT INTO {table_name} VALUES('
        for i in range(len(data.columns)):
            insert_query += '?,'
        insert_query = insert_query[:-1] + ')'
        self.cursor.executemany(insert_query, tuple_data)
        self.conn.commit()
        return True
        
    def delete(self, name):
        pass
        
    def close(self):
        self.conn.close()

    
class Influx_writer():

    def __init__(self, config):
        self.db = influx_db(config)

    def file_exists(self, file):
        if file:
            return self.db.table_exists(file)
        else:
            return False

    def read(self, file):
        query = "SELECT * FROM {}".format(file)
        if 'read_time' in self.db_config.keys() and self.db_config['read_time']:
            query += f" time > now() - {self.db_config['read_time']}"
        return self.db.read(query)

    def write(self, data, file):
        self.db.write(data, file)

    def close(self):
        pass


class influx_db():

    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.user = config.get('user', None)
        self.password = config.get('password', None)
        self.token = config.get('token', None)
        self.database = config['database']
        self.measurement = config['measurement']
        self.tags = config['tags']
        self.client = self.get_client()

    def table_exists(self, name):
        query = f"SHOW MEASUREMENTS ON {self.database}"
        result = self.client(query)
        for measurement in result['measurements']:
            if measurement['name'] == name:
                return True
        return False
        
    def read(self, query)->pd.DataFrame:
        cursor = self.client.query(query)
        points = cursor.get_points()
        my_list=list(points)
        df=pd.DataFrame(my_list)
        return df

    def get_client(self):
        headers = None
        if self.token:
            headers={"Authorization": self.token} 
        client = InfluxDBClient(self.host,self.port,self.user, self.password,headers=headers)
        databases = client.get_list_database()
        databases = [x['name'] for x in databases]
        if self.database not in databases:
            client.create_database(self.database)
        return InfluxDBClient(self.host,self.port,self.user,self.password,self.database,headers=headers)

    def write(self,data, measurement=None):
        if isinstance(data, pd.DataFrame):
            sorted_col = data.columns.tolist()
            sorted_col.sort()
            data = data[sorted_col]
            data = data.to_dict(orient='records')
        if not measurement:
            measurement = self.measurement
        for row in data:
            if 'time' in row.keys():
                p = '%Y-%m-%dT%H:%M:%S.%fZ'
                time_str = datetime.strptime(row['time'], p)
                del row['time']
            else:
                time_str = None
            if 'model_ver' in row.keys():
                self.tags['model_ver']= row['model_ver']
                del row['model_ver']
            json_body = [{
                'measurement': measurement,
                'time': time_str,
                'tags': self.tags,
                'fields': row
            }]
            self.client.write_points(json_body)

    def delete(self, name):
        pass
        
    def close(self):
        self.client.close()
