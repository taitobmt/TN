#!/usr/bin/env python
import MySQLdb
db = MySQLdb.connect("localhost", "monitor", "password", "USER")
curs=db.cursor()
# note that I'm using triplle quotes for formatting purposes
# you can use one set of double quotes if you put the whole string on one line
try:
    sql = """INSERT INTO Detail(ID, Name, tdate, ttime) values('1','Tu', CURRENT_DATE(), NOW())"""
    curs.execute(sql)
    sql1 = """INSERT INTO Detail(ID, Name, tdate, ttime) values('2','Thang', CURRENT_DATE(), NOW())"""
    curs.execute(sql1)
    db.commit()
    print ("Data committed")
except:
    print ("Error: the database is being rolled back")
    db.rollback()