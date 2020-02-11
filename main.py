from scene import Scene2D


def main():
    scene = Scene2D("data/rocket.xml", "data/traj.xml")
    scene.show_atmosphere()
    scene.sim("data/optim.xml")     # "data/optim.xml"
    scene.show_rocket_results()


if __name__ == '__main__':
    main()
else:
    print('Error: no entering point!')
    exit(-1)
