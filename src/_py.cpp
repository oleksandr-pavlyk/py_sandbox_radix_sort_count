#include <sycl/sycl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dpctl4pybind11.hpp"
#include "parallel_sort_count.hpp"
#include <vector>
#include <cstdint>
#include <utility>
#include "utils/type_dispatch.hpp"

class my_krn;

namespace td_ns = dpctl::tensor::type_dispatch;

std::pair<sycl::event, sycl::event>
py_radix_sort_count(
   dpctl::tensor::usm_ndarray vals,
   dpctl::tensor::usm_ndarray counts,
   size_t segments,
   size_t block_size,
   std::uint32_t radix_offset,
   std::vector<sycl::event> &depends)
{
  if (vals.get_ndim() != 1 || counts.get_ndim() != 1) {
    throw py::value_error("Input arrays must be vectors");
  }

  sycl::queue exec_q = vals.get_queue();
  if (!dpctl::utils::queues_are_compatible(exec_q, {counts})) {
    throw py::value_error("Incompatible allocation queues: can not deduce execution placement");
  }

  if (!vals.is_c_contiguous() || !counts.is_c_contiguous()) {
    throw py::value_error("Input arrays must be C-contiguous");
  }

  int vals_typenum = vals.get_typenum();
  int counts_typenum = counts.get_typenum();

  auto const &array_types = td_ns::usm_ndarray_types();
  int vals_typeid = array_types.typenum_to_lookup_id(vals_typenum);
  int counts_typeid = array_types.typenum_to_lookup_id(counts_typenum);

  constexpr int int64_typeid = static_cast<int>(td_ns::typenum_t::INT64);
  if (vals_typeid != int64_typeid || counts_typeid != int64_typeid) {
    throw py::value_error("Input arrays have unsupported data types");
  }

  auto e = radix_sort::count_submit<class my_krn, 4, true>(
     exec_q,
     segments,
     block_size,
     radix_offset,
     vals.get_data<std::int64_t>(),
     vals.get_size(),
     counts.get_data<std::int64_t>(),
     depends);

  sycl::event keep_arg_alive_ht_ev =
    dpctl::utils::keep_args_alive(exec_q, {vals, counts}, {e});

  return std::make_pair(keep_arg_alive_ht_ev, e);
}


PYBIND11_MODULE(_radix, m) {

  m.def("sort_count", &py_radix_sort_count,
	"sort_count(vals, counts, segments, block_size, radix_offset, depends)",
	py::arg("vals"),
	py::arg("counts"),
	py::arg("segments"),
	py::arg("block_size"),
	py::arg("radix_offset"),
	py::arg("depends") = py::list()
  );
}
