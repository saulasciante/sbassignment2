import matplotlib.pyplot as plt
import time
from main import runDetector


def plotGraph(x, y, title, xlabel, ylabel, savePath=None):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.plot(x, y)

    if savePath:
        plt.savefig(savePath)

    plt.show()
    plt.clf()


if __name__ == '__main__':
    timePlot = []
    scaleFactorPlot = []
    neighboursPlot = []
    scaleFactors = [1.01, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    nOfNeighbors = [1, 2, 3, 4, 5, 6, 7, 8]

    for scaleFactor in scaleFactors:
        startTime = time.time()
        scaleFactorPlot.append(runDetector(scaleFactor, 3))
        timePlot.append(time.time() - startTime)

    for n in nOfNeighbors:
        neighboursPlot.append(runDetector(1.01, n))

    plotGraph(scaleFactors, timePlot, 'Execution time based on scale factor', 'Scale factor', 'Execution time', savePath='time.png')
    plotGraph(scaleFactors, scaleFactorPlot, 'Detection accuracy based on scale factor', 'Scale factor', 'Detection accuracy', savePath='factors.png')
    plotGraph(nOfNeighbors, neighboursPlot, 'Detection accuracy based on minimal number of neighbours', 'Minimal number of neighbours', 'Detection accuracy', savePath='neighbours.png')
