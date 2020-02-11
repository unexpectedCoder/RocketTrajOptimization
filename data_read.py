from htypes import Coordinates, Compartment, Constraints, Parameters, StartConds
from engine import Engine
from traj import *
from rocket import Rocket2D

from lxml import etree
import numpy as np


def create_rocket(atmo, xml_rocket: str, xml_traj: str):
    """
    Creates rocket's object using XML data file.
    :param atmo: atmosphere's model
    :type atmo: Atmosphere
    :param xml_rocket: rocket's XML file's path
    :param xml_traj: trajectory's XML file's path
    """
    s_conds, compartments, constraints, engine, params = read_rocket_xml(xml_rocket)
    traj = read_traj_xml(xml_traj)
    return Rocket2D(atmo, s_conds, params, constraints, engine, compartments, traj)


def read_rocket_xml(xml_path: str):
    """
    Reads rocket's XML data file.
    :param xml_path: rocket's XML file's path
    :return: StartConds, list[Compartment], Constraints, Engine, Parameters
    """
    root = etree.ElementTree(file=xml_path)
    if check_rocket_xml(root) is False:
        print("Error in <scene>: invalid rocket's XML file structure!")
        exit(-1)
    # Starting conditions
    node = root.find('StartConds')
    coords = Coordinates(np.array([float(node.get('x')) * 1e3, float(node.get('y')) * 1e3]))
    s_conds = StartConds(float(node.get('t')), coords, float(node.get('V')), float(node.get('mu')),
                         np.deg2rad(float(node.get('Theta'))))
    # Compartments
    node = root.find('Compartments')
    compartments = [Compartment(child.get('name'), float(child.get('mass'))) for child in node]
    # Constraints
    node = root.find('Constraints')
    constraints = Constraints(float(node.get('max_dist')) * 1e3, float(node.get('max_height')) * 1e3, float(node.get('max_ny')))
    # Engine
    eta, imp1, p0, p_outlet, k, break_mu = [], [], [], [], [], []
    node = root.find('Engine')
    sub_nodes = node.findall('Mode')
    for sub_node in sub_nodes:
        for child in sub_node:
            if child.tag == 'eta':
                eta.append(float(child.text))
            elif child.tag == 'imp1':
                imp1.append(float(child.text))
            elif child.tag == 'p0':
                p0.append(float(child.text) * 1e6)
            elif child.tag == 'p_outlet':
                p_outlet.append(float(child.text) * 1e6)
            elif child.tag == 'k':
                k.append(float(child.text))
            elif child.tag == 'break_mu':
                break_mu.append(float(child.text))
            else:
                continue
    engine = Engine(np.array(eta), np.array(imp1), np.array(p0), np.array(p_outlet), np.array(k), np.array(break_mu))
    # Parameters
    node = root.find('Parameters')
    params = Parameters(float(node.get('icx')), float(node.get('beta')), float(node.get('q_m')), float(node.get('lambda_l')))
    return s_conds, compartments, constraints, engine, params


def check_rocket_xml(root):
    """
    Checks rocket's XML file structure.
    :param root: root XML handle
    :type root: etree._Element
    :return: True (correct structure) or False (incorrect structure)
    """
    # Starting conditions
    node = root.find('StartConds')
    if node is None:
        return False
    # Compartments
    node = root.find('Compartments')
    if node is None:
        return False
    # Constraints
    node = root.find('Constraints')
    if node is None:
        return False
    # Engine
    node = root.find('Engine')
    if node is None:
        return False
    sub_nodes = node.findall('Mode')
    for sub_node in sub_nodes:
        if sub_node is None:
            return False
    # Parameters
    node = root.find('Parameters')
    if node is None:
        return False
    return True


def read_traj_xml(xml_path: str):
    """
    Reads trajectory's XML data file.
    :param xml_path: XML file's path
    :return: list of trajectory's parts
    """
    root = etree.ElementTree(file=xml_path)
    if check_traj_xml(root) is False:
        print("Error in <scene>: invalid trajectory's XML file structure!")
        exit(-1)
    nodes = root.findall('Part')
    parts = []
    for node in nodes:
        name = node.get('name')
        t = node.get('t')
        theta = node.get('Theta')
        over = node.get('overload_share')
        events = node.get('events')

        t = float(t) if t != 'None' else None
        theta = np.deg2rad(float(theta)) if theta != 'None' else None
        over = float(over) if over != 'None' else None
        events = events.split(' ') if events != 'None' else [None]

        parts.append(TrajPart(name, t, theta, over, events))
    return Traj(parts)


def check_traj_xml(root):
    """
    Checks trajectory's XML data file structure.
    :param root: XML root handle
    :type root: etree._Element
    :return: True (correct structure) or False (incorrect structure)
    """
    if root.find('Part') is None:
        return False
    return True


def init_optim_range(xml_path: str):
    """
    Reads optimizer's XML data file and init the range of optimization values.
    :param xml_path: XML file's path
    :return: list of optimization's values ranges (min/max)
    """
    root = etree.ElementTree(file=xml_path)
    if check_optim_xml(root) is False:
        print("Error in <scene>: invalid optimizer's XML file structure!")
        exit(-1)
    node = root.find('min')
    x_min = [np.deg2rad(float(node.get('Theta'))), float(node.get('k_t')), float(node.get('h_march')) * 1e3]
    node = root.find('max')
    x_max = [np.deg2rad(float(node.get('Theta'))), float(node.get('k_t')), float(node.get('h_march')) * 1e3]
    return list(zip(x_min, x_max))


def check_optim_xml(root):
    """
    Checks optimizer's XML data file structure.
    :param root: XML root handle
    :type root: etree._Element
    :return: True (correct structure) or False (incorrect structure)
    """
    if root.find('min') is None:
        return False
    if root.find('max') is None:
        return False
    return True
