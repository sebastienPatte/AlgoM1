import sys
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import random as rdm
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser


stein_file = "test.stp"

#librairies :
# python3 -m pip install steinlib
# python3 -m pip install networkx
# doc nx https://networkx.github.io/documentation/stable/index.html

# draw a graph in a window
def print_graph(graph,terms=None,sol=None):

    pos=nx.kamada_kawai_layout(graph)
    
    nx.draw(graph,pos)
    if (not (terms is None)):
        nx.draw_networkx_nodes(graph,pos, nodelist=terms, node_color='r')
    if (not (sol is None)):
        nx.draw_networkx_edges(graph,pos, edgelist=sol, edge_color='r')
    plt.show()
    
    return


# verify if a solution is correct and evaluate it
def eval_sol(graph,terms,sol):

    if (len(sol)==0):
        print("Error: empty solution")
        return -1

    graph_sol = nx.Graph()
    for (i,j) in sol:
        graph_sol.add_edge(i,j,weight=graph[i][j]['weight'])

    # is sol a tree
    if (not (nx.is_tree(graph_sol))):
        print ("Error: the proposed solution is not a tree")
        return -1

    # are the terminals covered
    for i in terms:
        if not i in graph_sol:
            print ("Error: a terminal is missing from the solution")
            return -1

    # cost of solution
    cost = graph_sol.size(weight='weight')

    return cost



# compute an approximate solution to the steiner problem
def approx_steiner(graph,terms):
    res = []    
    Gt = nx.complete_graph(terms)
    
    #calcul de tout les plus court chemins
    path = dict(nx.all_pairs_shortest_path(graph))

    #pour chaque combinaison de 2 noeuds
    for (i,j) in it.combinations(terms,2):
        #on met le poids au plus court chemin dans Gt
        Gt.add_edge(i,j,weight=nx.path_weight(graph,path[i][j],weight="weight"))

    #arbre couvrant minimum de Gt        
    A = list(nx.minimum_spanning_tree(Gt).edges(data=True))
    
    for i in range(len(A)):
        pcc = path[A[i][0]][A[i][1]];
        for j in range(len(pcc)-1):
            
            pair = (pcc[j], pcc[j+1])    
            
            res.append(pair)
    
    print_graph(Gt,terms)
    
    return res 


def init(graph):
    Es = []
    for i in range(len(nx.edges(graph))):
        Es.append(rdm.randint(0, 1))
    return Es

# def recuit(graph, terms, nb_iter):
#     for i in nb_iter:
#         newI
        
        
def neighbor(sol,graph):
    i = rdm.randint(0, len(sol)-1)
    sol[i] = 1-sol[i]
    return sol
    # for i in range(len(Es)-1):
    #     if rand < proba:
    #         Es[i] = 1 - Es[i]
    
def eval(sol, graph, terms):
    
    
# class used to read a steinlib instance
class MySteinlibInstance(SteinlibInstance):

    def __init__(self):
        self.my_graph = nx.Graph()
        self.terms = []


    def terminals__t(self, line, converted_token):
        self.terms.append(converted_token[0])

    def graph__e(self, line, converted_token):
        e_start = converted_token[0]
        e_end = converted_token[1]
        weight = converted_token[2]
        self.my_graph.add_edge(e_start,e_end,weight=weight)



if __name__ == "__main__":
    my_class = MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        # print_graph(graph,terms)
        # sol=approx_steiner(graph,terms)
        # print_graph(graph,terms,sol)
        # print(eval_sol(graph,terms,sol))
        
        I = init(graph,terms)
        print("initial Es : ",I)
        print("neighbor   : ",neighbor(I, graph, terms))


