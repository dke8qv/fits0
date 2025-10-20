#include "TRandom2.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TGClient.h"
#include "TStyle.h"
#include <iostream>
#include "TF1.h"
#include "TFitResult.h"
#include "TPaveText.h"
#include "TROOT.h"
using namespace std;
using TMath::Log;

//parms
const double xmin=1;
const double xmax=20;
const int npoints=12;
const double sigma=0.2;

double f(double x){
  const double a=0.5;
  const double b=1.3;
  const double c=0.5;
  return a+b*Log(x)+c*Log(x)*Log(x);
}

void getX(double *x){
  double step=(xmax-xmin)/npoints;
  for (int i=0; i<npoints; i++){
    x[i]=xmin+i*step;
  }
}

void getY(const double *x, double *y, double *ey){
  static TRandom2 tr(0);
  for (int i=0; i<npoints; i++){
    y[i]=f(x[i])+tr.Gaus(0,sigma);
    ey[i]=sigma;
  }
}

void leastsq(){
  double x[npoints];
  double y[npoints];
  double ey[npoints];
  getX(x);
  getY(x,y,ey);
  auto tg = new TGraphErrors(npoints,x,y,0,ey);
  tg->Draw("alp");
}

int main(int argc, char **argv){
  TApplication theApp("App", &argc, argv); // init ROOT App for displays

  // ******************************************************************************
  // ** this block is useful for supporting both high and std resolution screens **
  UInt_t dh = gClient->GetDisplayHeight()/2; // fix plot to 1/2 screen height
  //UInt_t dw = gClient->GetDisplayWidth();
  UInt_t dw = 1.1*dh;
  // ******************************************************************************
  gStyle->SetOptStat(0); // turn off histogram stats box

  gROOT->SetBatch(kTRUE);

  TCanvas *tc = new TCanvas("c1","Sample dataset",dw,dh);
  double lx[npoints];
  double ly[npoints];
  double ley[npoints];
  getX(lx);
  getY(lx,ly,ley);
  auto tgl = new TGraphErrors(npoints,lx,ly,0,ley);
  tgl->SetTitle("Pseudoexperiment;x;y"); // An example of one pseudo experiment
  tgl->Draw("alp");
  tc->Draw();

  TH2F *h1 = new TH2F("h1","Parameter b vs a;a;b",100,0,2,100,0,2);
  TH2F *h2 = new TH2F("h2","Parameter c vs a;a;c",100,0,2,100,0,2);
  TH2F *h3 = new TH2F("h3","Parameter c vs b;b;c",100,0,2,100,0,2);
  TH1F *h4 = new TH1F("h4","reduced chi^2;;frequency",100,0,3);

  const int Nexp = 2000;

  TH1F *ha = new TH1F("ha","Parameter a; a; frequency",100,0,2);
  TH1F *hb = new TH1F("hb","Parameter b; b; frequency",100,0,2);
  TH1F *hc = new TH1F("hc","Parameter c; c; frequency",100,0,2);
  TH1F *hchi2 = new TH1F("hchi2","chi^2;;frequency",100,0,60);

  TF1 fitf("fitf","[0] + [1]*log(x) + [2]*log(x)*log(x)", xmin, xmax);

  double sumChi2 = 0.0;
  double sumChi22 = 0.0;

  for (int ie=0; ie<Nexp; ++ie){
    getX(lx);
    getY(lx,ly,ley);
    TGraphErrors tg(npoints,lx,ly,0,ley);
    TFitResultPtr res = tg.Fit(&fitf,"QSN");
    double a = fitf.GetParameter(0);
    double b = fitf.GetParameter(1);
    double c = fitf.GetParameter(2);
    double chi2 = res->Chi2();
    int ndf = res->Ndf();

    ha->Fill(a);
    hb->Fill(b);
    hc->Fill(c);
    hchi2->Fill(chi2);
    h1->Fill(a,b);
    h2->Fill(a,c);
    h3->Fill(b,c);
    if (ndf>0) h4->Fill(chi2/ndf);

    sumChi2  += chi2;
    sumChi22 += chi2*chi2;
  }

  int ndf = npoints - 3;
  double meanChi2 = sumChi2 / Nexp;
  double varChi2  = (sumChi22 / Nexp) - meanChi2*meanChi2;
  if (varChi2 < 0) varChi2 = 0;
  double stdChi2  = TMath::Sqrt(varChi2);
  double expMean  = ndf;
  double expStd   = TMath::Sqrt(2.0*ndf);

  cout << "Mean chi2 = " << meanChi2 << "  Expected = " << expMean << endl;
  cout << "Std chi2  = " << stdChi2  << "  Expected = " << expStd  << endl;
  cout << "Mean reduced chi2 ~ " << meanChi2 / ndf << endl;

  TCanvas *tc1 = new TCanvas("c_dist","parameter/chi2 distributions",200,200,dw,dh);
  tc1->Divide(2,2);
  tc1->cd(1); ha->Draw();
  tc1->cd(2); hb->Draw();
  tc1->cd(3); hc->Draw();
  tc1->cd(4); hchi2->Draw();
  tc1->Draw();

  TCanvas *tc2 = new TCanvas("c2","my study results",200,200,dw,dh);
  tc2->Divide(2,2);
  tc2->cd(1); h1->Draw("colz");
  tc2->cd(2); h2->Draw("colz");
  tc2->cd(3); h3->Draw("colz");
  tc2->cd(4); h4->Draw();
  tc2->Draw();

  tc1->Print("LSQFitROOT.pdf(");
  tc2->Print("LSQFitROOT.pdf");

  TCanvas *tc3 = new TCanvas("c3", "Comments / Analysis", 200, 200, dw, dh);
tc3->cd();
TPaveText *pt = new TPaveText(0.05, 0.05, 0.95, 0.95, "NDC");
pt->SetTextSize(0.03);
pt->AddText("Comments:");
{
    TString s; s.Form("- Mean chi2 = %.2f  (Expected ~ %d)", meanChi2, ndf);
    pt->AddText(s);
}
{
    TString s; s.Form("- Std chi2 = %.2f  (Expected ~ %.2f)", stdChi2, expStd);
    pt->AddText(s);
}
{
    TString s; s.Form("- Mean reduced chi2 = %.2f  (Expected ~ 1)", meanChi2/ndf);
    pt->AddText(s);
}
pt->AddText("- More data points -> narrower parameter spreads.");
pt->AddText("- Larger sigma -> wider parameter spreads and higher chi2.");
pt->Draw();
tc3->Print("LSQFitROOT.pdf)");


  return 0;
}


