import mysql.connector
from mysql.connector import Error

class DatabaseData:
    def __init__(self, no, nama, tanggal_lahir, tinggi_badan, berat_badan, 
                 panjang_tangan, panjang_kaki, panjang_paha, lebar_paha, lebar_dada, date):
        self.no = no
        self.nama = nama
        self.tanggal_lahir = tanggal_lahir
        self.tinggi_badan = tinggi_badan
        self.berat_badan = berat_badan
        self.panjang_tangan = panjang_tangan
        self.panjang_kaki = panjang_kaki
        self.panjang_paha = panjang_paha
        self.lebar_paha = lebar_paha
        self.lebar_dada = lebar_dada
        self.date = date

class DatabaseHandler:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect_to_mysql(self):
        try:
            return mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def upload_to_mysql(self, data: DatabaseData, table_name: str = "pengukuran"):
        conn = None  # Inisialisasi variabel conn
        try:
            conn = self.connect_to_mysql()
            cursor = conn.cursor()  # Membuka cursor secara manual

            # Buat tabel jika belum ada
            create_table_query = (
                f"CREATE TABLE IF NOT EXISTS {table_name} ("
                f"no INT AUTO_INCREMENT PRIMARY KEY, "
                f"nama VARCHAR(100), "
                f"tanggal_lahir DATE, "
                f"tinggi_badan FLOAT, "
                f"berat_badan FLOAT, "
                f"panjang_tangan FLOAT, "
                f"panjang_kaki FLOAT, "
                f"panjang_paha FLOAT, "
                f"lebar_paha FLOAT, "
                f"lebar_dada FLOAT, "
                f"date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            cursor.execute(create_table_query)

            # Masukkan data ke tabel
            insert_query = (
                f"INSERT INTO {table_name} "
                f"(nama, tanggal_lahir, tinggi_badan, berat_badan, panjang_tangan, "
                f"panjang_kaki, panjang_paha, lebar_paha, lebar_dada, date) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )
            cursor.execute(insert_query, (
                data.nama, data.tanggal_lahir, data.tinggi_badan, data.berat_badan,
                data.panjang_tangan, data.panjang_kaki, data.panjang_paha,
                data.lebar_paha, data.lebar_dada, data.date
            ))

            # Simpan perubahan
            conn.commit()
            print("Data berhasil dimasukkan ke database.")
        except Error as e:
            print(f"Error saat mengunggah ke MySQL: {e}")
        finally:
            # Tutup cursor dan koneksi
            if conn is not None:
                if 'cursor' in locals():
                    cursor.close()  # Pastikan cursor ditutup
                conn.close()
                print("Koneksi MySQL ditutup.")


# Contoh Penggunaan
if __name__ == "__main__":
    # Konfigurasi koneksi database
    db_handler = DatabaseHandler(host="localhost", user="root", password="", database="database-antropometri")

    # Data contoh
    sample_data = DatabaseData(
        no=None,
        nama="John Doe",
        tanggal_lahir="1990-01-01",
        tinggi_badan=175.5,
        berat_badan=70.3,
        panjang_tangan=60.2,
        panjang_kaki=90.1,
        panjang_paha=50.3,
        lebar_paha=25.4,
        lebar_dada=40.5,
        date="2025-01-08"
    )

    # Unggah data
    db_handler.upload_to_mysql(sample_data)
