module HodgkinHuxley
export HH, rk_update!

using Base: @kwdef
using Parameters: @unpack

"""
パラメータ定義
"""
@kwdef struct HHParameter{FT}
    Cm::FT = 1.0
    gNa::FT = 120.0
    gK::FT = 36.0
    gL::FT = 0.3
    # 標準的な値に
    ENa::FT = 50.0
    EK::FT = -77.0
    EL::FT = -54.387

    # 以下は追加のパラメータ(可塑性などを想定？)
    tr::FT = 0.5  # ms
    td::FT = 8.0  # ms
    invtr::FT = 1.0 /  tr
    invtd::FT = 1.0 /  td
    v0::FT  = -20.0    # mV
end

"""
変数をまとめた構造体
"""
@kwdef mutable struct HH{FT}
    param::HHParameter{FT} = HHParameter{FT}()
    N::UInt16
    # 初期条件
    v::Vector{FT} = fill(-65.0, N)
    m::Vector{FT} = fill(0.05,  N)
    h::Vector{FT} = fill(0.6,   N)
    n::Vector{FT} = fill(0.32,  N)
    r::Vector{FT} = zeros(N)
end

"""
微分方程式 df/dt を返す関数
(v, m, h, n) を入力とし、(dm, dh, dn, dv) を返す
"""
function df(param::HHParameter, v, m, h, n, Ie)
    @unpack Cm, gNa, gK, gL, ENa, EK, EL = param
    # 温度スケーリング
    temp = 3^((15 - 6.3)/10)

    # m
    αm = 0.1 * (v + 40.0) / (1.0 - exp(-0.1 * (v + 40.0)))
    βm = 4.0 * exp(- (v + 65.0) / 18.0)
    dm = (αm * (1.0 - m) - βm * m) * temp

    # h
    αh = 0.07 * exp(-0.05 * (v + 65.0))
    βh = 1.0 / (1.0 + exp(-0.1 * (v + 35.0)))
    dh = (αh * (1.0 - h) - βh * h) * temp

    # n
    αn = 0.01 * (v + 55.0) / (1.0 - exp(-0.1 * (v + 55.0)))
    βn = 0.125 * exp(-0.0125 * (v + 65.0))
    dn = (αn * (1.0 - n) - βn * n) * temp

    # v
    dv = (1 / Cm) * (
        Ie
        - gNa * m^3 * h * (v - ENa)
        - gK  * n^4 * (v - EK)
        - gL  * (v - EL)
    )

    return dm, dh, dn, dv
end

"""
1点分の (v,m,h,n) に対して 4次のRunge-Kutta法で 1ステップ更新
"""
function rk4(param::HHParameter, func, v, m, h, n, dt, Ie)
    # k1
    dm1, dh1, dn1, dv1 = func(param, v, m, h, n, Ie)

    # k2
    dm2, dh2, dn2, dv2 = func(
        param,
        v + 0.5*dt*dv1,
        m + 0.5*dt*dm1,
        h + 0.5*dt*dh1,
        n + 0.5*dt*dn1,
        Ie
    )

    # k3
    dm3, dh3, dn3, dv3 = func(
        param,
        v + 0.5*dt*dv2,
        m + 0.5*dt*dm2,
        h + 0.5*dt*dh2,
        n + 0.5*dt*dn2,
        Ie
    )

    # k4
    dm4, dh4, dn4, dv4 = func(
        param,
        v + dt*dv3,
        m + dt*dm3,
        h + dt*dh3,
        n + dt*dn3,
        Ie
    )

    # 合成
    v_new = v + (dt/6.0)*(dv1 + 2*dv2 + 2*dv3 + dv4)
    m_new = m + (dt/6.0)*(dm1 + 2*dm2 + 2*dm3 + dm4)
    h_new = h + (dt/6.0)*(dh1 + 2*dh2 + 2*dh3 + dh4)
    n_new = n + (dt/6.0)*(dn1 + 2*dn2 + 2*dn3 + dn4)

    return v_new, m_new, h_new, n_new
end

"""
RK4を用いて (v,m,h,n) を更新し, rはEulerで更新する
"""
function rk_update!(variable::HH, param::HHParameter, Ie::Vector, dt)
    @unpack N, v, m, h, n, r = variable
    @unpack tr, td, invtr, invtd, v0 = param

    @inbounds for i = 1:N
        # (v,m,h,n) を RK4 で更新
        v_new, m_new, h_new, n_new =
            rk4(param, df, v[i], m[i], h[i], n[i], dt, Ie[i])

        v[i] = v_new
        m[i] = m_new
        h[i] = h_new
        n[i] = n_new

        # r のみ Euler
        r[i] += dt * (
            (invtr - invtd)*(1.0 - r[i]) / (1.0 + exp(-(v[i] - v0)))
            - r[i]*invtd
        )
    end
end

end  # module HodgkinHuxley
