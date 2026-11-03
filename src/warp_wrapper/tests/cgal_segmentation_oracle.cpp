#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/mesh_segmentation.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Mesh = CGAL::Surface_mesh<Kernel::Point_3>;

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: cgal_segmentation_oracle MESH K LAMBDA\n";
    return 2;
  }
  Mesh mesh;
  if (!CGAL::IO::read_polygon_mesh(argv[1], mesh) || !CGAL::is_triangle_mesh(mesh)) {
    std::cerr << "failed to read triangular mesh\n";
    return 3;
  }
  const std::size_t k = std::stoul(argv[2]);
  const double lambda = std::stod(argv[3]);
  auto sdf = mesh.add_property_map<Mesh::Face_index, double>("f:sdf", 0.0).first;
  auto raw = mesh.add_property_map<Mesh::Face_index, double>("f:raw_sdf", 0.0).first;
  auto segment = mesh.add_property_map<Mesh::Face_index, std::size_t>("f:segment", 0).first;
  CGAL::sdf_values(mesh, raw, 2.0 / 3.0 * CGAL_PI, 25, false);
  for (Mesh::Face_index face : mesh.faces()) sdf[face] = raw[face];
  const auto range = CGAL::sdf_values_postprocessing(mesh, sdf);
  const std::size_t count = CGAL::segmentation_from_sdf_values(
      mesh, sdf, segment, k, lambda);
  std::cout << std::setprecision(17);
  std::cout << "segments " << count << "\n";
  std::cout << "sdf_range " << range.first << " " << range.second << "\n";
  for (Mesh::Face_index face : mesh.faces()) {
    std::cout << face.idx() << " " << sdf[face] << " " << segment[face]
              << " " << raw[face] << "\n";
  }
  return 0;
}
