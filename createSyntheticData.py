import random, time, string
random.seed(time.time())

for index in range(0, 1000):
	filename = "./test300/random" + str(index) + ".txt"
	f = open(filename, "w")

	count = 0
	end_year = 2010
	end_month = 12
	end_time = 1000000
	year = random.randint(2000, end_year)
	month = random.randint(1, end_month)
	time = random.randint(1, end_time)
	
	#create increasing item num with random existence
	for i in range(0, 1000):
		if i % random.randint(1, 3) == 0:
			continue
		count = count + 1
		if count == 300:
			break
		time = random.randint(1, end_time)
		'''to_next = random.choice([True, False])
		if to_next == True:
			time = random.randint(time, end_time)
			#year = random.randint(year, end_year)
			#month = random.randint(month, end_month)
		else:
			time = random.randint(1, time)
			#year = random.randint(2000, year)
			#month = random.randint(1, month)
		'''
		#time = str(year) + str(month).zfill(2)
		item = i + 1
		price = random.randint(1000, 100000)
		f.write(str(item) + " " +  str(time) + " " +  str(price) + "\n")

	'''
	#create random
	for i in range(0, 260):
		year = str(random.randint(2000, 2010))
		month = str(random.randint(1, 12))
		time = year + month.zfill(2)

		#category = random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase)
		item = random.randint(1, 150)
		price = random.randint(1000, 100000)
		#price = int(price / 1000)
		#price = price * 1000
		f.write(str(item) + " " + time + " " + str(price)+"\n")
	'''
	f.close()

'''
# item name with capital character
import random, time, string, csv
random.seed(time.time())

f = open('user1.csv', 'w', encoding='utf-8', newline='\n')
wr = csv.writer(f)

for i in range(0, 10000):
	year = str(random.randint(2000, 2020))
	month = str(random.randint(1, 12))
	time = year + month.zfill(2)

	category = random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase)

	price = random.randint(1000, 1000000)
	price = int(price / 1000)
	price = price * 1000
	wr.writerow([category, time, price])'''