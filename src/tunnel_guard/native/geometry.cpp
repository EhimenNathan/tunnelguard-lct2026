// Геометрическое ядро Gauss–Newton: билинейное поле расстояний и нормальные уравнения.
// Без fast-math: сохраняем численную эквивалентность эталону NumPy/Numba.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
namespace py = pybind11;
using A = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I = py::array_t<long long, py::array::c_style | py::array::forcecast>;

py::tuple gn(A px_, A py_, A pz_, I k_, A f_, A sw_, A y_, A z_, A head_, double roll,
             A field_, double trunc, double l0, double h0, double res, double core_w,
             double core_h0, double core_h1, double vmax_h, double vmin_l, double vbed_h,
             double vroof_h, int n) {
    auto px=px_.unchecked<1>(), yy=py_.unchecked<1>(), zz=pz_.unchecked<1>();
    auto k=k_.unchecked<1>(); auto f=f_.unchecked<1>(), sw=sw_.unchecked<1>();
    auto y=y_.unchecked<1>(), z=z_.unchecked<1>(), head=head_.unchecked<1>();
    auto F=field_.unchecked<2>();
    const auto N=px.shape(0);
    if(n<2 || y.shape(0)<n || z.shape(0)<n || head.shape(0)<n ||
       yy.shape(0)!=N || zz.shape(0)!=N || k.shape(0)!=N || f.shape(0)!=N || sw.shape(0)!=N ||
       res<=0 || trunc<=0) throw py::value_error("invalid geometry kernel dimensions or scale");
    for(py::ssize_t q=0;q<N;++q)
        if(k(q)<0 || k(q)>=n-1) throw py::value_error("geometry node index out of bounds");
    std::vector<A> out;
    for(int i=0;i<6;++i) out.emplace_back(n);
    auto jy0=out[0].mutable_unchecked<1>(), jy1=out[1].mutable_unchecked<1>(), ry=out[2].mutable_unchecked<1>();
    auto jz0=out[3].mutable_unchecked<1>(), jz1=out[4].mutable_unchecked<1>(), rz=out[5].mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for(int i=0;i<n;++i) jy0(i)=jy1(i)=ry(i)=jz0(i)=jz1(i)=rz(i)=0.;
        for(py::ssize_t q=0;q<N;++q) {
            const int kk=static_cast<int>(k(q)); const double ff=f(q), gg=1.-ff;
            const double cs=1./std::sqrt(1.+head(kk)*head(kk));
            const double yc=y(kk)*gg+y(kk+1)*ff, zc=z(kk)*gg+z(kk+1)*ff;
            const double l=(yy(q)-yc)*cs, h=zz(q)-zc-roll*l;
            const double u=(l-l0)/res-.5, v=(h-h0)/res-.5;
            double val=trunc, du=0., dv=0.;
            if(u>=0 && u<F.shape(0)-1 && v>=0 && v<F.shape(1)-1) {
                int i=static_cast<int>(u), j=static_cast<int>(v);
                double a=u-i,b=v-j, f00=F(i,j),f10=F(i+1,j),f01=F(i,j+1),f11=F(i+1,j+1);
                val=f00*(1-a)*(1-b)+f10*a*(1-b)+f01*(1-a)*b+f11*a*b;
                du=((f10-f00)*(1-b)+(f11-f01)*b)/res;
                dv=((f01-f00)*(1-a)+(f11-f10)*a)/res;
            }
            double w=sw(q);
            if(trunc<1.) { double t=std::fmin(val/trunc,1.); w*=1.-t*t; }
            const double al=std::abs(l);
            if(al<core_w && h>core_h0-.3 && h<core_h1) w=0.;
            const double r=w*val, gy=w*(-cs)*(du-roll*dv);
            const bool vm=(h<vmax_h && (al>vmin_l || h<vbed_h)) || h>vroof_h;
            double J0=gy*gg,J1=gy*ff;
            jy0(kk)+=J0*J0; jy0(kk+1)+=J1*J1; jy1(kk)+=J0*J1;
            ry(kk)+=J0*r; ry(kk+1)+=J1*r;
            if(vm) {
                double gz=w*(-dv); J0=gz*gg;J1=gz*ff;
                jz0(kk)+=J0*J0;jz0(kk+1)+=J1*J1;jz1(kk)+=J0*J1;
                rz(kk)+=J0*r;rz(kk+1)+=J1*r;
            }
        }
    }
    return py::make_tuple(out[0],out[1],out[2],out[3],out[4],out[5]);
}
PYBIND11_MODULE(_geometry_native, m) { m.def("gn", &gn); }
