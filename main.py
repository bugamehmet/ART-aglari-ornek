import numpy as np

N = 4  #
M = 4
benzerlik_esik_degeri = 0.8

w_F2den_F1e = np.ones((M, N))

w_F1den_F2ye = np.ones((N, M))

X = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1]
])

sinif_vektorleri = np.eye(M)

for i in range(N):
    x = X[:, i]

    F2_ciktilar = np.dot(w_F1den_F2ye[i, :], w_F2den_F1e)

    kazanan = np.argmax(F2_ciktilar)

    s1 = np.sum(x)
    s2 = np.sum(sinif_vektorleri[kazanan])
    benzerlik = s2 / s1

    if benzerlik >= benzerlik_esik_degeri:
        w_F1den_F2ye[i, kazanan] = 1
        w_F2den_F1e[kazanan, i] = 1
    else:
        F2_ciktilar[kazanan] = 0
        ikinci_kazanan = np.argmax(F2_ciktilar)

        if ikinci_kazanan == kazanan:
            M += 1
            yeni_sinif_vektoru = np.zeros(M)
            yeni_sinif_vektoru[M - 1] = 1
            w_F1den_F2ye = np.hstack((w_F1den_F2ye, np.zeros((N, 1))))
            w_F2den_F1e = np.vstack((w_F2den_F1e, np.zeros((1, N))))
            w_F1den_F2ye[i, kazanan] = 1
            w_F2den_F1e[kazanan, i] = 1
        else:
            kazanan = ikinci_kazanan
            w_F1den_F2ye[i, kazanan] = 1
            w_F2den_F1e[kazanan, i] = 1

            print(f"Örnek {i + 1} için Sınıf: {kazanan}")

            # Sonuç
            print("Ağırlık Matrisi F1'den F2'ye:")
            print(w_F1den_F2ye)

            print("Ağırlık Matrisi F2'den F1'e:")
            print(w_F2den_F1e)
