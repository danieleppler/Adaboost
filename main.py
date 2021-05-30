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
	n = 20
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
			r = Rule(p1, p2)
			Rules.append(r)

	empirical_error_test = 0
	empirical_error_train = 0

	iteration = 1
	for i in range(n):

		f.write("iteration number: " + str(iteration) + '\n')
		iteration = iteration + 1
		Train, Test = sklearn.model_selection.train_test_split(Points, train_size=75, test_size=75)

		for p in Train:
			p.weight = (1 / 75)

		best_rules = []

		for t in range(8):
			best_Rule = 0
			min_error = sys.maxsize
			for h in Rules:
	#			if h not in best_rules:
					et = 0
					for p in Train:
						hx = check_claissified_point(p, h)
						if hx == 0:
							et = et + p.weight
					if et < min_error:
						best_Rule = h
						min_error = et


			if min_error >= 0.5:
				min_error = 1 - min_error

			num = ((1-min_error)/min_error)
			alpha = (1/2) * np.log(num)
			r = Rule(best_Rule.p1,best_Rule.p2)
			r.weight = alpha
			best_rules.append(r)

			z = 0

			for p in Points:
					hx = check_claissified_point(p, best_Rule)
					if hx == 0:
						hx = -1
					p.weight = p.weight * pow(math.e, (-1 * best_Rule.weight * hx))
					z = z + p.weight

			for p in Points:
					p.weight = (p.weight / z)

		rule_number = 1
		for h in best_rules:
			ee_test_rule = 0
			ee_train_rule = 0
			for p in Train:
				hx = sign(best_rules, p, (rule_number))
				if hx == -1:
					ee_train_rule = ee_train_rule + 1
			for p in Test:
				hx = sign(best_rules, p,(rule_number))
				if hx == -1:
					ee_test_rule = ee_test_rule + 1
			f.write("success rate of rule " + str(rule_number) + " on train : " + str(round((100 - (100 * (ee_train_rule/75))),2))
			        + "%        success rate of rule " + str(rule_number) + " on test : " + str(round((100 - (100 * (ee_test_rule/75))),2)) + "%" + '\n')
			empirical_error_test = empirical_error_test + (100 - (100 * (ee_test_rule/75)))
			empirical_error_train = empirical_error_train + (100 - (100 * (ee_train_rule/75)))
			rule_number = rule_number + 1

		f.write('\n \n')
	f.write("AVG of success rate on train : " + str((empirical_error_train)/(8*n)) +
	        "     AVG of succes rate on test : " + str((empirical_error_test)/(8*n)))


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

def sign(best_rules,p,k):
	sum=0
	for i in range(k):
		hx = check_claissified_point(p, best_rules[i])
		if hx == 0:
			hx = -1
		sum = sum + best_rules[i].weight * hx
	if sum > 0:
		return 1
	else:
		return -1




if __name__ == '__main__':
	main()