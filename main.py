

import random
from Rule import Rule
from Point import Point
from itertools import product
import sys
import numpy as np
import math
import sklearn.model_selection

def main():
	file = open("rectangle.txt","r")
	f = open("output.txt", "a")
	line = file.readline()
	Points= []
	Rules= []
	Train = []
	Test = []
	while line != "":
		x = line[0:3]
		y = line[5:8]
		type = line[10]
		if type == '-':
			type = -1
		p = Point(x, y, type)
		Points.append(p)
		line = file.readline()

	combinations_object = list(product(Points, repeat=2))
	for i in range(len(combinations_object)):
		p1 = combinations_object[i][0]
		p2 = combinations_object[i][1]
		if p1 != p2:
			r = Rule(p1,p2)
			Rules.append(r)

	empirical_error_test = 0
	empirical_error_train = 0

	for i in range(30):
		for p in Points:
			p.weight = 1 / 75

		Train, Test = sklearn.model_selection.train_test_split(Points, train_size=75, test_size=75)

		best_rules = []

		for t in range(8):
			minRule = Rule(0,0)
			min = sys.maxsize
			for h in Rules:
				if h not in best_rules:
					et = 0
					for p in Train:
						hx = check_claissified_point(p,h)
						if hx == 0:
							hx = 1
						else:
							hx = 0
						et = et + p.weight * hx
					if et < min:
						minRule = h
						min = et

			minRule.weight = (1/2) * np.log((1-min)/min)
			best_rules.append(minRule)

			z = 0

			for p in Train:
				hx = check_claissified_point(p,minRule)
				p.weight = p.weight * pow(math.e, (-1 * minRule.weight * hx))
				z = z + p.weight

			for p in Train:
				p.weight = p.weight / z


		for h in best_rules:
			ee_test_rule = 0
			ee_train_rule = 0
			for p in Test:
				hx = check_claissified_point(p,h)
				if hx == 0:
					ee_test_rule = ee_test_rule + 1
			for p in Train:
				hx = check_claissified_point(p,h)
				if hx == 0:
					ee_train_rule = ee_train_rule + 1
			f.write("emprical error on train : " + str(ee_train_rule)
			        + "        emprical error on test : " + str(ee_test_rule) + '\n')
			empirical_error_test = empirical_error_test + ee_train_rule
			empirical_error_train = empirical_error_train + ee_test_rule

		f.write('\n \n')

	f.write("AVG of emprical error on train : " + str((empirical_error_train/100)) +
	        "     AVG of emprical error on test : " + str((empirical_error_test/100)))


def check_claissified_point(p,h):
	type = 0
	hx = 0
	# checking if this rule classified the point as 1 or -1 using  Cross product
	v1 = [float(h.p1.x) - float(h.p2.x), float(h.p1.y) - float(h.p2.y)]  # Vector 1
	v2 = [float(h.p1.x) - float(p.x), float(h.p1.y) - float(p.y)] # Vector 1
	xp = v1[0] * v2[1] - v1[1] * v2[0]
	if xp > 0:
		type = 1
	elif xp < 0:
		type = -1
	else:
		type = random.choice([-1, 1])
	if p.type == type:
		hx = 1
	else:
		hx = 0
	return hx




if __name__ == '__main__':
	main()