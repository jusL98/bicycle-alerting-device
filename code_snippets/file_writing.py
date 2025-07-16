from datetime import datetime

file = open("log","a")
file.write(datetime.now().strftime('\n%Y-%m-%d %H:%M:%S') + ' - Object Detected')

print(datetime.now().strftime('\n%Y-%m-%d %H:%M:%S') + ' - Object Detected')
file.close()
