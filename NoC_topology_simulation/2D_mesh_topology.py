import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()

M = 4
N = 4

for i in range(M):
    for j in range(N):
        G.add_node((i, j))

for i in range(M):
    for j in range(N):
        if i+1 < M:
            G.add_edge((i, j), (i+1, j))
        if j+1 < N:
            G.add_edge((i, j), (i, j+1))

pos = {}
for i in range(M):
    for j in range(N):
        pos[(i, j)] = (i, M - 1 - j)


nx.draw(G, pos=pos)
plt.savefig("graph.png")