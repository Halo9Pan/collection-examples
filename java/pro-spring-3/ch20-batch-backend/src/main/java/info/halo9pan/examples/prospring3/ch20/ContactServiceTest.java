/**
 * Created on Dec 21, 2011
 */
package info.halo9pan.examples.prospring3.ch20;

import info.halo9pan.examples.prospring3.ch20.domain.Contact;
import info.halo9pan.examples.prospring3.ch20.domain.Hobby;
import info.halo9pan.examples.prospring3.ch20.service.ContactService;

import java.util.List;

import org.springframework.context.support.GenericXmlApplicationContext;

/**
 * @author Clarence
 *
 */
public class ContactServiceTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		GenericXmlApplicationContext ctx = new GenericXmlApplicationContext();
		ctx.load("classpath:app-context.xml");
		ctx.refresh();

		System.out.println("App context initialized successfully");

		ContactService contactService = ctx.getBean("contactService", ContactService.class);

		List<Contact> contacts = contactService.findAll();

		for (Contact contact : contacts) {
			System.out.println(contact);
		}

	}

}
