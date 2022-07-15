#include "polyscope/polyscope.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "polyscope/curve_network.h"

struct SimData
{
    SimData(int segs, double sep);
    
    int segs;    
    Eigen::VectorXd restq;
    Eigen::SparseMatrix<double> M;
    Eigen::SparseMatrix<double> Minv;
    double segLength;
};

struct SimState
{
    Eigen::VectorXd q;
    Eigen::VectorXd qdot;
    Eigen::VectorXd lambda;
};

SimData::SimData(int segs, double chi) : segs(segs)
{
    int nverts = segs+1;
    restq.resize(3*nverts);
    
    double len = 1.022;
    segLength = len / double(segs);
    double sep = chi * len;
    double totM = 0.0208;
    std::vector<Eigen::Triplet<double> > Mcoeffs;
    std::vector<Eigen::Triplet<double> > Minvcoeffs;
    for(int i=0; i<nverts; i++)
    {
        double mass = totM / double(segs);
        if(i==0 || i == nverts-1)
            mass /= 2.0;
        for(int j=0; j<3; j++)
        {
            Mcoeffs.push_back({3*i+j, 3*i+j, mass});
            Minvcoeffs.push_back({3*i+j, 3*i+j, 1.0/mass});
        }
    }
    M.resize(3*nverts, 3*nverts);
    M.setFromTriplets(Mcoeffs.begin(), Mcoeffs.end());
    Minv.resize(3*nverts, 3*nverts);
    Minv.setFromTriplets(Minvcoeffs.begin(), Minvcoeffs.end());
    
    double aguess = 1.0;
    while(2.0 * aguess * std::sinh(sep/(2.0 * aguess)) < len)
        aguess *= 0.5;
    double alow = aguess;
    aguess = 1.0;
    while(2.0 * aguess * std::sinh(sep/(2.0 * aguess)) > len)
    {
        aguess *= 2.0;
    }
    double ahigh = aguess;
    
    while(ahigh - alow > 1e-6)
    {
        double amid = 0.5*(ahigh+alow);
        if(2.0 * amid * std::sinh(sep/(2.0 * amid)) > len)
            alow = amid;
        else
            ahigh = amid;
    }
    for(int i=0; i<nverts; i++)
    {
        double s = -len/2.0 + len * i / double(nverts);
        double x = alow * std::asinh(s/alow);
        double y = alow * std::cosh(x/alow);
        restq[3*i+0] = x;
        restq[3*i+1] = y;
        restq[3*i+2] = 0;
    }
    
    double arclen = 0;
    for(int i=0; i<segs; i++)
    {
        arclen += (restq.segment<3>(3*(i+1)) - restq.segment<3>(3*i)).norm();
    }
    std::cout << arclen << std::endl;
    
    double H = (restq.segment<3>(3*segs) - restq.segment<3>(0)).norm();
    std::cout << H << std::endl;
}

void g(Eigen::VectorXd &q, int segs, double restLen, Eigen::VectorXd &g, std::vector<Eigen::Triplet<double> > *dg)
{
    g.resize(segs+3);
    g.segment<3>(0) = q.segment<3>(0);
    for(int i=0; i<segs; i++)
    {
        Eigen::Vector3d p1 = q.segment<3>(3*i);
        Eigen::Vector3d p2 = q.segment<3>(3*(i+1));
        g[3+i] = (p2-p1).squaredNorm() - restLen*restLen;
    }
    
    if(dg)
    {
        for(int i=0; i<3; i++)
        {
            dg->push_back({i, i, 1.0});
        }
        for(int i=0; i<segs; i++)
        {
            Eigen::Vector3d p1 = q.segment<3>(3*i);
            Eigen::Vector3d p2 = q.segment<3>(3*(i+1));
            Eigen::Vector3d coeff = 2.0 * (p2-p1);
            for(int j=0; j<3; j++)
            {
                dg->push_back({3+i, 3*i+j, -coeff[j]});
                dg->push_back({3+i, 3*(i+1)+j, coeff[j]});
            }
        }        
    }
}

void takeStep(const SimData &simdata, SimState &state, double h)
{
    constexpr double gravityg = -9.8;
    constexpr double newtonTol = 1e-6;
    
    Eigen::VectorXd lambdaguess = state.lambda;
    int nconstraints = simdata.segs + 3;
    if(lambdaguess.size() != nconstraints)
    {
        lambdaguess.resize(nconstraints);
        lambdaguess.setZero();
    }
    
    int nverts = simdata.restq.size()/3;
    Eigen::VectorXd F(3*nverts);
    F.setZero();
    for(int i=0; i<nverts; i++)
    {
        F[3*i+1] = gravityg;
    }    
    F = simdata.M * F;
    
    Eigen::VectorXd qip1 = state.q + h * state.qdot;
    Eigen::VectorXd qunc = qip1 + h * state.qdot + h * h * simdata.Minv * F;
    
    std::vector<Eigen::Triplet<double> > dgatq;
    std::vector<Eigen::Triplet<double> > dgatqguess;
    Eigen::VectorXd gatq;
    g(qip1, nverts-1, simdata.segLength, gatq, &dgatq);
    Eigen::VectorXd f;
    Eigen::SparseMatrix<double> Dgatq(nconstraints, 3*nverts);
    Dgatq.setFromTriplets(dgatq.begin(), dgatq.end());
    Eigen::VectorXd qguess = qunc - h * h * Dgatq.transpose() * lambdaguess;
    g(qguess, nverts-1, simdata.segLength, f, &dgatqguess);
    
    std::vector<Eigen::Triplet<double> > Icoeffs;
    for(int i=0; i<nconstraints; i++)
    {
        Icoeffs.push_back({i,i,1.0});
    }
    Eigen::SparseMatrix<double> I(nconstraints, nconstraints);
    I.setFromTriplets(Icoeffs.begin(), Icoeffs.end());
    
    double reg = 1e-6;
    
    while(f.norm() > newtonTol)
    {
        std::cout << f.transpose() << std::endl;
        std::cout << lambdaguess.transpose() << std::endl;
        std::cout << "fnorm is now " << f.norm() << std::endl;
        Eigen::SparseMatrix<double> Dgatqguess(nconstraints, 3*nverts);
        Dgatqguess.setFromTriplets(dgatqguess.begin(), dgatqguess.end());
        
        while(true)
        {
            Eigen::SparseMatrix<double> H = reg * I - h * h * Dgatqguess * Dgatq.transpose();
            Eigen::SparseLU<Eigen::SparseMatrix<double> > solver(H);
            Eigen::VectorXd deltalambda = solver.solve(-f);
            Eigen::VectorXd newlambdaguess = lambdaguess + deltalambda;
            Eigen::VectorXd qguess = qunc - h * h * Dgatq.transpose() * newlambdaguess;
            Eigen::VectorXd newf;
            g(qguess, nverts-1, simdata.segLength, newf, NULL);            
            if(newf.norm() > f.norm())
            {            
                reg *= 2.0;
                std::cout << "solve failed, " << newf.norm() << " > " << f.norm() << ", reg now " << reg << std::endl;
            }
            else
            {
                reg *= 0.5;            
                std::cout << "accepted " << deltalambda.transpose() << std::endl;
                lambdaguess = newlambdaguess;
                dgatqguess.clear();
                g(qguess, nverts-1, simdata.segLength, f, &dgatqguess);
                break;
            }
        }

    }
}

void configToMatrix(const Eigen::VectorXd &q, Eigen::MatrixXd &M)
{
    int dim = q.size()/3;
    M.resize(dim, 3);
    for(int i=0; i<dim; i++)
    {
        M.row(i) = q.segment<3>(3*i);
    }
}

int main(int argc, char *argv[])
{
    polyscope::init();
    
    SimData simd(100,0.25);
    
    Eigen::MatrixXd renderrestq;
    configToMatrix(simd.restq, renderrestq);
    
    polyscope::registerCurveNetworkLine("rest rod", renderrestq);
    
    std::cout << renderrestq << std::endl;
    
    SimState state;
    state.q = simd.restq;
    state.qdot = state.q;
    state.qdot.setZero();
    
    double dt = 0.01;
    
    takeStep(simd, state, dt);
    
    
    polyscope::show();   
}
