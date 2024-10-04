import sqlite3

#Connect to Sqlite3
connection = sqlite3.connect("Student.db")

#Create a Cursor object to Insert Records, Create table, Retriever, Response
cursor = connection.cursor()

#Create Table
table_info = """
CREATE TABLE STUDENT(NAME VARCHAR(15), CLASS VARCHAR(20), SECTION VARCHAR(10), MARKS INT);
"""
cursor.execute(table_info)

#Insert Some Records
cursor.execute("""INSERT INTO STUDENT VALUES("VAMSHI", "DATA SCIENCE", "C",85)""")
cursor.execute("""INSERT INTO STUDENT VALUES("KRISHNA", "MACHINE LEARNING", "A",90)""")
cursor.execute("""INSERT INTO STUDENT VALUES("SHIN CHAN", "AI", "C",88)""")
cursor.execute("""INSERT INTO STUDENT VALUES("BHEEM", "DATA SCIENCE", "B",80)""")

#Display the all records
print("The Inserted Records are:")
data = cursor.execute("""SELECT * FROM STUDENT""")
for row in data:
    print(row)

#Close Connections
connection.commit()
connection.close()
