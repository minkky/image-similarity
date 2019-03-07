import random, time, string
random.seed(time.time())

for index in range(0, 1000):
	filename = "./dataset/random" + str(index) + ".txt"
	f = open(filename, "w")

	for i in range(0, 10000):
		year = str(random.randint(2000, 2020))
		month = str(random.randint(1, 12))
		time = year + month.zfill(2)

		#category = random.choice(string.ascii_uppercase) + random.choice(string.ascii_uppercase)
		item = random.randint(1, 300)
		price = random.randint(1000, 1000000)
		price = int(price / 1000)
		price = price * 1000
		f.write(str(item) + " " + time + " " + str(price)+"\n")
	f.close()

'''import random, time, string, csv
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