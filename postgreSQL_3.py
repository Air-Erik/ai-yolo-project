import psycopg

with psycopg.connect('dbname=AI_Database user=ayrapetov_es \
password=1111') as conn:
    with conn.cursor() as cur:

        cur.execute(
            'INSERT INTO new_table (class, class_id, x_1, y_1, x_2, y_2, \
            percent) VALUES (%s, %s, %s, %s, %s, %s, %s)',
            ('well', 10, 0.1, 1.1, 20.1, 30.1, 80.25))

        conn.commit()

print('yes')
