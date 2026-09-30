# -*- coding: utf-8 -*-
"""O localizador do número sobrevive ao ruído da organização.

É ele que transforma o span do extrator neural na chave do índice — se costurar
uma palavra vizinha ao número, a citação real vira `inventada` em silêncio.
"""
import pytest

from gama.normalizar import nucleo_numerico


@pytest.mark.parametrize("trecho,chave", [
    # exemplos de ruído do próprio dev set da organização
    ("AgInt no RESP 21737l8 - SP", "2173718"),
    ("R.Esp. n°  1.45g.779-MA", "1459779"),
    ("EDcl no AgInt no Recurso Especial Nº 170076O (SP)", "1700760"),
    ("Recl. n° 6G.838/ BA", "66838"),
    ("AgRg no RESP 1.528.4S5/ RJ", "1528455"),
    ("Reclamação n.  44-\n.921 (PE)", "44921"),
    ("ED no AgR-REspe No  0600316-4920206160182", "6003164920206160182"),
    ("REspe. n° 0600216-46.2020-\n.6.14.0022", "6002164620206140022"),
    ("TST-ED-E-ED-ARR-1099-66.2011.5.02. 0251", "10996620115020251"),
    # prefixo que o ruído transformou em "dígito" não pode entrar no número
    ("Rc1 88178/RS", "88178"),
    ("EDcl n0 AgInt no  REsp 188q127/SE", "1889127"),
    ("AgInt no Recurso Especia1 nº 1.620.021/PR", "1620021"),
    ("RSE 7000082- 11.2026.7.00.0000/RS", "70000821120267000000"),
])
def test_nucleo_resiste_ao_ruido(trecho, chave):
    assert nucleo_numerico(trecho) == chave
