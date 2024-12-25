import mysql.connector

class DatabaseData:
    def __init__(self, height, weight):
        self.height = height
        self.weight = weight

class MysqlClient:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect_to_mysql(self):
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def upload_to_mysql(self, data: DatabaseData, table_name:str = "body_size"):
        conn = self.connect_to_mysql()
        cursor = conn.cursor()

        # Create table if not exists
        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (id INT AUTO_INCREMENT PRIMARY KEY, height FLOAT, weight FLOAT)"
        cursor.execute(create_table_query)

        cursor.execute(f"INSERT INTO {table_name} (height, weight) VALUES (%s, %s)", (data.height, data.weight))

        # Commit changes and close connection
        conn.commit()
        conn.close()


# if __name__ == "__main__":
#     mysql_client = MysqlClient("192.168.1.101", "a", "test_password", "bodydata")
#     mysql_client.upload_to_mysql(DatabaseData(1.70, 80.00), "body_size")