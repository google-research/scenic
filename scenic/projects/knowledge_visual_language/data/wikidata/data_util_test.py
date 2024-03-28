# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Extract WikiGraph."""
from absl import logging
from absl.testing import absltest
from scenic.projects.knowledge_visual_language.data.wikidata import data_util


class ClipModelsTest(absltest.TestCase):
  """Single Test based on a dummy input."""

  def test_graph_extraction(self):
    """Tests whether it could correctly extract 2-hop graph."""
    rel_list = [
        ["Q1066517", "r1", "Qdummy"],
        ["Q35581", "r2", "Qdummy"],
        ["Q2339039", "r2", "Qdummy"],
        ["Q753878", "r2", "Qdummy"],
    ]
    logging.info("Construct KG Graph from edge list.")
    kg_graph = data_util.construct_kg_graph(rel_list)
    logging.info("Extract 2-hop subgraphs within wiki-pages.")
    demo_data = self.demo_input()
    for d in demo_data:
      in_context_ents = {ent: True for ent in d["link"] if ent != "UNK"}
      graph = data_util.extract_2hop_graph(in_context_ents, kg_graph)
      print(graph)
      logging.info(len(graph))

  def demo_input(self):
    d_list = [{
        "txt": "yukigassen\n is a snowball fighting-competition from "
               "japan. today there are annual tournaments in "
               "s\u014dbetsu, hokkaid\u014d in japan, kemij\u00e4rvi in"
               " finland, vard\u00f8 in norway, murmansk in russia, "
               "mount buller, victoria in australia, lule\u00e5 in "
               "sweden, anchorage in alaska, aparan in armenia, jasper,"
               " alberta and saskatoon, saskatchewan in canada.\nthe "
               "word consists of the japanese words yuki (snow) and "
               "kassen (battle) with rendaku. hence \"yukigassen\" "
               "means snow battle, but is a common term for 'snowball "
               "fight' in japanese.\n......",
        "link": {
            "Q1066517": [[17, 46], [483, 497]],
            "Q35581": [[106, 114]],
            "Q744704": [[125, 134]],
            "Q33": [[138, 145]],
            "Q108983": [[147, 152], [1720, 1725], [1907, 1912], [2089, 2094]],
            "Q1763": [[164, 172]],
            "Q984117": [[184, 206]],
            "Q26268": [[221, 226], [1534, 1539], [1015, 1020], [1175, 1180],
                       [1363, 1368]],
            "Q39450": [[238, 247]],
            "Q797": [[251, 257]],
            "Q39618": [[259, 265]],
            "Q399": [[269, 276]],
            "Q999429": [[278, 293]],
            "Q10566": [[298, 321]],
            "Q5287": [[358, 366]],
            "Q3943791": [[379, 383], [444, 448], [1627, 1631]],
            "Q178561": [[397, 403]],
            "Q1192464": [[410, 417]],
            "Q1035213": [[744, 760]],
            "Q131647": [[964, 969], [1118, 1123], [1305, 1310], [1477, 1482],
                        [1669, 1674], [1865, 1870], [2047, 2052], [925, 930]],
            "Q406039": [[1029, 1039], [1189, 1199], [1377, 1387], [1548, 1558],
                        [1734, 1744], [1922, 1926], [2104, 2108]],
            "Q847956": [[1052, 1064], [1211, 1223], [1404, 1416], [1584, 1596],
                        [1766, 1778], [1951, 1957], [2146, 2152]],
            "Q873364": [[1073, 1085], [1243, 1255], [1429, 1441], [1608, 1620],
                        [1808, 1820], [1975, 1981], [2175, 2181]],
            "Q44853": [[1745, 1753]],
            "Q218082": [[0, 10], [426, 436], [522, 532], [704, 714], [831,
                                                                      841]],
            "UNK": [[97, 104], [698, 725], [1040, 1051], [1200, 1210],
                    [1442, 1452], [1597, 1607], [1224, 1242], [1256, 1280],
                    [1559, 1583], [1388, 1403], [1621, 1640], [1779, 1795],
                    [1821, 1826], [1959, 1964], [1928, 1940], [1983, 2006]]
        },
        "entity": "Yukigassen"
    }, {
        "txt":
            "harry hyams\nharry john hyams (2 january 1928 \u2013 19 december "
            "2015) was a british millionaire who initially made his money as a"
            " speculative property (real estate) developer. he was best known "
            "as the developer of the centre point office building in "
            "london.\n......",
        "link": {
            "Q2339039": [[213, 225], [961, 973]],
            "Q149787": [[283, 289]],
            "Q19186": [[291, 300]],
            "Q753878": [[633, 642]],
            "Q743535": [[1139, 1146]],
            "Q43747844": [[1301, 1310]],
            "Q7290153": [[1388, 1402], [1566, 1580]],
            "Q7743416": [[1584, 1600]],
            "Q5669913": [[0, 11], [12, 28]]
        },
        "entity": "Harry Hyams"
    }]
    return d_list


if __name__ == "__main__":
  absltest.main()
