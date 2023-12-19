import pandas as pd
import psycopg
from psycopg import sql

#Извлечение данных из базы данных и запись их в pandas DF
query = sql.SQL('SELECT * FROM "AutoCAD_Topo_v7"')

data = []

with psycopg.connect('dbname=ai_database user=ayrapetov_es \
password=1111') as conn:
    with conn.cursor() as cur:
        cur.execute(query)
        cur.fetchone()
        for res in cur:
            new_row = {'x_1': res[0],
                       'y_1': res[1],
                       'x_2': res[2],
                       'y_2': res[3],
                       'percent': res[4],
                       'file_name': res[5],
                       'class': res[6],
                       'class_id': res[7]}
            data.append(new_row)

df = pd.DataFrame(data)
print(df.head(10))