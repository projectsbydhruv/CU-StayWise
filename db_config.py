# db_config.py

def get_db_params():
    return {
        "host": "localhost",
        "port": 5433,          # from docker inspect
        "dbname": "postgres",  # default DB since POSTGRES_DB not set
        "user": "postgres",    # default user
        "password": "123"      # from POSTGRES_PASSWORD
    }


def get_connection_string():
    params = get_db_params()
    return (
        f"postgresql+psycopg2://{params['user']}:{params['password']}"
        f"@{params['host']}:{params['port']}/{params['dbname']}"
    )
