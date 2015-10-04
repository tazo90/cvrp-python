import sys, random
from math import sqrt, ceil
from pickle import *
from PIL import Image, ImageDraw, ImageFont
import time

from pygooglechart import Chart
from pygooglechart import SimpleLineChart
from pygooglechart import Axis

num_cities = 10
route = []
demand = 1
capacity = 3
COLORS = [(255,0,0), (0,255,0), (0,0,255)]
img_file = "test.png"
delay_time = 0.21
start_score = 0
max_cities_to_visit = int(ceil( ((num_cities-1) / float(capacity)) ))
best_distances = []
#city with index 0 is depot

def draw_plot(data, generation):
	# Set the vertical range from 0 to 100
	max_y = data[0]

	# Chart size of 200x125 pixels and specifying the range for the Y axis
	chart = SimpleLineChart(600, 325, y_range=[0, max_y])

	# Add the chart data
	"""
	data = [
	    32, 34, 34, 32, 34, 34, 32, 32, 32, 34, 34, 32, 29, 29, 34, 34, 34, 37,
	    37, 39, 42, 47, 50, 54, 57, 60, 60, 60, 60, 60, 60, 60, 62, 62, 60, 55,
	    55, 52, 47, 44, 44, 40, 40, 37, 0,0,0,0,0,0,0
	]
	"""	
	chart.add_data(data)

	# Set the line colour to blue
	chart.set_colours(['0000FF'])

	# Set the vertical stripes
	chart.fill_linear_stripes(Chart.CHART, 0, 'CCCCCC', 0.2, 'FFFFFF', 0.2)

	# Set the horizontal dotted lines
	chart.set_grid(0, 25, 5, 5)

	# The Y axis labels contains 0 to 100 skipping every 25, but remove the
	# first number because it's obvious and gets in the way of the first X
	# label.
	left_axis = range(0, max_y + 1, 25)
	left_axis[0] = ''
	chart.set_axis_labels(Axis.LEFT, left_axis)

	# X axis labels
	chart.set_axis_labels(Axis.BOTTOM, [str(x) for x in xrange(1, generation+1)][::14])
	chart.download('plot.png')


def get_route():
	global route
	lst = [i for i in xrange(num_cities)]
	for i in xrange( int(ceil(num_cities / float(capacity))) ):
		track_route = []
		t_length = len(lst)-1 if len(lst)-1 < int(capacity) else capacity
		for j in xrange(t_length):			
			choice = random.choice(lst[1:])
			lst.remove(choice)
			track_route.append(choice)
		route.append(track_route)
	return route

def draw_image(img_file, coords, total_route, start_score, best_score, gen_num):	
	"""
	Draw cities from coords
	"""
	img = Image.new("RGB", (800,600), color=(255,255,255))
	font = ImageFont.load_default()
	d = ImageDraw.Draw(img)
	num_cities = len(coords)

	depot = coords[0]
	d.text( depot, "depot", font=font, fill=(32, 32, 32)) 
	
	k = 0
	for track_route in total_route:					
		color = COLORS[k]
		k = (k + 1) % len(COLORS)		
		for i in xrange(len(track_route)):			
			j = (i+1) % len(track_route)
			city_from = track_route[i]
			city_to = track_route[j]
			x1, y1 = coords[city_from]
			x2, y2 = coords[city_to]

			d.line((int(x1), int(y1), int(x2), int(y2)), fill=color)
			d.text((int(x1) + 7, int(y1) - 5), str(city_from), font=font, fill=(32,32,32))		
		
	for x, y in coords:
		x, y = int(x), int(y)
		d.ellipse( (x - 5, y - 5, x + 5, y + 5), outline=(0,0,0), fill=(196,196,196) )		

	d.text((10, 10), str("Generation no: %d" % gen_num), font=font, fill=(32,32,32))
	d.text((10, 25), str("Cities no: %d" % num_cities), font=font, fill=(32,32,32))
	d.text((10, 40), str("Best distance: %.2f km" % best_score), font=font, fill=(32,32,32))
	d.text((10, 55), str("Start distance: %.2f km" % start_score), font=font, fill=(32,32,32))
	d.text((10, 70), str("Vehicle capacity: %d" % capacity), font=font, fill=(32,32,32))
	d.text((10, 85), str("Client demands: %d" % demand), font=font, fill=(32,32,32))

	del d
	img.save(img_file, "PNG")
	#print "The plot was savet in the %s file." % (img_file,)

def get_distance_matrix(coords):
	"""
	Returns distance matrix of a given (x,y) coords
	"""
	matrix = {}
	for i, (x1, y1) in enumerate(coords):
		for j, (x2, y2) in enumerate(coords):
			dx = x1 - x2
			dy = y1 - y2
			dist = sqrt(dx*dx + dy*dy)
			matrix[i, j] = dist
	return matrix


def get_cities_coords(num_cities, xmax=800, ymax=600):
	"""
	Calculate random position of a city (x,y - coord)
	"""
	coords = []
	for i in range(num_cities):
		x = random.randint(0, xmax)
		y = random.randint(0, ymax)
		coords.append( (float(x), float(y)) )
	return coords

def eval_func(chromosome):
	""" 
	The evaluation function 
	"""
	global cm
	return get_route_length(cm, chromosome)

cm = []
coords = []

class Individual:
	score = 0	
	depot = 0 # has always val 00

	def __init__(self, chromosome=None, depot=0):
		self.chromosome = chromosome or self._makechromosome()		
		self.score = 0	
		self.depot = depot		
		self.split_chromosome = self.split_route_on_capacity_with_depot()		

	def _makechromosome(self):
		"""
		Makes a chromosome from randomly selected alleles
		"""
		chromosome = [self.depot]
		lst = [i for i in xrange(1,num_cities)]
		for i in xrange(1,num_cities):
			choice = random.choice(lst)
			lst.remove(choice)
			chromosome.append(choice)
		return chromosome

	def evaluate(self):	
		"""
		Calculates length of a route for current individual
		"""		
		self.score = self.get_route_length()

	def crossover(self, other):
		"""
		Cross two parents and returns created child's
		"""
		left, right = self._pickpivots()
		p1 = Individual()
		p2 = Individual()

		c1 = [c for c in self.chromosome[1:] if c not in other.chromosome[left:right+1]]
		p1.chromosome = [self.depot] + c1[:left] + other.chromosome[left:right+1] + c1[left:]
		c2 = [c for c in other.chromosome[1:] if c not in self.chromosome[left:right+1]]
		p2.chromosome = [other.depot] + c2[:left] + self.chromosome[left:right+1] + c2[left:]
		
		#print '====== ', p1, p2		
		return p1, p2

	def mutate(self):
		""" 
		Swap two elements 
		"""
		left, right = self._pickpivots()
		self.chromosome[left], self.chromosome[right] = self.chromosome[right], self.chromosome[left]

	def _pickpivots(self):
		"""
		Returns random left, right pivots 
		"""
		left = random.randint(1, num_cities - 2)
		right = random.randint(left, num_cities - 1)
		return left, right

	def copy(self):
		twin = self.__class__(self.chromosome[:])
		twin.score = self.score
		return twin

	def split_route_on_capacity_with_depot(self):		
		"""
		Split route of cities [1,2,3,4] to routes depending on capacity 
		"""
		podzial = []
		total_podzialy = 0	

		while total_podzialy < (num_cities-1):
			length = random.randint(1, max_cities_to_visit)						

			if length + total_podzialy < num_cities:
				total_podzialy += length
				podzial.append(length)	
			
		step = 0		
		self.split_routes = []
		for i,city in enumerate(podzial):
			route = [self.chromosome[0]] + self.chromosome[1+step:podzial[i]+step+1]
			step += podzial[i]
			self.split_routes.append(route)

		return self.split_routes

	def get_route_length(self):
		"""
		Returns the total length of the route
		"""
		total = 0		
		global cm
		
		for track_route in self.split_routes:
			for i in xrange(len(track_route)):
				j = (i + 1) % len(track_route)
				city_from = track_route[i]
				city_to = track_route[j]
				total += cm[city_from, city_to]

		return total

	def __repr__(self):
		return '<%s chromosome="%s" score=%s>' % (self.__class__.__name__, str(self.split_chromosome), self.score)


class Environment:
	size = 0
	def __init__(self, population=None, size=3, maxgenerations=2,\
				 newindividualrate=0.6, crossover_rate=0.90,\
				 mutation_rate=0.1):
		self.size = size
		self.population = self._makepopulation()
		self.maxgenerations = maxgenerations
		self.newindividualrate = newindividualrate
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.generation = 0
		self.minscore = sys.maxint
		self.minindividual = None

	def _makepopulation(self):
		return [Individual() for i in xrange(0, self.size)]

	def run(self):
		for i in xrange(1, self.maxgenerations + 1):
			print 'Generation no: ' + str(i) + '\n'
			for j in range(0, self.size):
				self.population[j].evaluate()								
				#print 'first ', self.population[j].split_routes, self.population[j].score
				curscore = self.population[j].score				
				if curscore < self.minscore:
					#print 'set min ', self.population[j].score
					self.minscore = curscore
					self.minindividual = self.population[j].copy()						
			print 'Best individual: ', self.minindividual, ' ', id(self.minindividual)

			#best_distances.append(self.minindividual.score)			

			#draw_plot(best_distances, i)

			#print self.minindividual.chromosome
			if i == 1:
				start_score = self.minindividual.score
			draw_image(img_file, coords, self.minindividual.split_chromosome, start_score, self.minindividual.score, i)
			time.sleep(delay_time)

			# crossover parents to create better child's
			if random.random() < self.crossover_rate:
				children = []
				# 60% total population will be crossover
				newindividual = int(self.newindividualrate * self.size )
				for i in xrange(0, newindividual):
					# select best parent to crossover					
					selected1 = self._selectrank()
					while True:
						selected2 = self._selectrank()									
						if selected1 != selected2:
							break

					parent1 = self.population[selected1]
					parent2 = self.population[selected2]					
					child1, child2 = parent1.crossover(parent2)
					child1.evaluate()
					child2.evaluate()
					#self.population = sorted(self.population, key=lambda p: p.score, reverse=True)

					set_child1, set_child2 = False, False
					
					if child1.score < self.population[0].score:						
						self.population.pop(0)
						self.population.append(child1)
						#print self.population
						set_child1 = True

					if child2.score < self.population[1].score:
						self.population.pop(1)
						self.population.append(child2)
						#print self.population
						set_child1 = True

					if not set_child1 and not set_child2:
						if child2.score < self.population[0].score:
							self.population.pop(0)
							self.population.append(child2)

						if child1.score < self.population[1].score:
							self.population.pop(1)			
							self.population.append(child1)												

			# mutation
			if random.random() < self.mutation_rate:
				selected = self._select()	# select some individual to mutate
				self.population[selected].mutate()

		#end loop
		for i in xrange(0, self.size):
			self.population[i].evaluate()
			curscore = self.population[i].score			
			if curscore < self.minscore:
				self.minscore = curscore
				self.minindividual = self.population[i].copy()				
				#print 'set min 2 ', self.minindividual, ' ', self.minindividual.score, ' ', id(self.minindividual)

		print '.................Result.................'
		print self.minindividual

	def _select(self):
		totalscore = 0
		for i in xrange(0, self.size):
			totalscore += self.population[i].score

		randscore = random.random() * (self.size - 1)
		addscore = 0
		selected = 0
		for i in xrange(0, self.size):
			addscore += (1 - self.population[i].score / totalscore)
			if addscore >= randscore:
				selected = i
				break
		return selected

	def _selectrank(self):
		return random.randint(0,self.size-1)

dist_matrix = []

def main_run():
	global cm, coords, num_cities

	#f = open('testy/miasta15.txt', 'r')
	#num_cities = len(f.readline().strip().split(';'))
	#f.seek(0)
	
	# get cities coords	
	num_cities = 40
	coords = get_cities_coords(num_cities)	
	cm = get_distance_matrix(coords)

	#for line in f:
	#	dist_matrix.append(map(int, line.strip().split(';')))	

	#num_cities = len(dist_matrix[0])	

	#matrix = {}
	#for i in xrange(0, num_cities):
	#	for j in xrange(0, num_cities):
	#		matrix[i, j] = dist_matrix[i][j]
	
	#cm = matrix
	ev = Environment(size=100, maxgenerations=300)
	ev.run()
	

if __name__ == '__main__':
	main_run()



