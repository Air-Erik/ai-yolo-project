import psycopg

with psycopg.connect('dbname=AI_Database user=ayrapetov_es \
password=1111') as conn:
    with conn.cursor() as cur:

        cur.execute(
            'INSERT INTO new_table (class, class_id, x_1, y_1, x_2, y_2, \
            percent) VALUES (%s, %s, %s, %s, %s, %s, %s)',
            ('well', 6, 20.1, 21.1, 2.1, 3.1, 80.25))

        cur.execute("SELECT * FROM new_table")
        cur.fetchone()

        for record in cur:
            print(record)

print('yes')
