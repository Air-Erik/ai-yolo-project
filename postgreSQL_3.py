import psycopg

with psycopg.connect('dbname=ai_database user=ayrapetov_es \
password=1111') as conn:
    with conn.cursor() as cur:

        cur.execute(
            'INSERT INTO train (class, class_id, x_1, y_1, x_2, y_2, \
            percent, file_name) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
            ('well', 10, 0.1, 1.1, 20.1, 30.1, 80.25, 'erik'))
        cur.close()

    conn.commit()

print('yes')
