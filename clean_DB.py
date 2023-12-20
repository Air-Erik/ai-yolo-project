import psycopg

with psycopg.connect('dbname=ai_database user=ayrapetov_es \
password=1111') as conn:
    with conn.cursor() as cur:

        cur.execute(
            'DELETE FROM "AutoCAD_Topo_v7"'
            )
        cur.close()

    conn.commit()
